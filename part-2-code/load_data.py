import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

TOKENIZER = T5TokenizerFast.from_pretrained("google-t5/t5-small", use_fast=True)

PAD_IDX = TOKENIZER.pad_token_id                
BOS_ID = TOKENIZER.pad_token_id       

MAX_SRC_LEN = 256
MAX_TGT_LEN = 256   # avoid truncating longer SQL queries



class T5Dataset(Dataset):
    def __init__(self, data_folder, split):
        """
        Data for T5 fine-tuning:
          - split in {"train","dev","test"}
          - For train/dev we have NL and SQL pairs
          - For test we only have NL
        We tokenize here so collate can just pad + build masks/shifts.
        """
        self.split = split
        self.items = []  # each item: {"enc_ids": LongTensor, "dec_ids": LongTensor or None}
        self.process_data(data_folder, split, TOKENIZER)


    def process_data(self, data_folder, split, tokenizer):
        if split == "train":
            nl_path  = os.path.join(data_folder, "train.nl")
            sql_path = os.path.join(data_folder, "train.sql")
            nl_list  = load_lines(nl_path)
            sql_list = load_lines(sql_path)
            assert len(nl_list) == len(sql_list), "train NL/SQL size mismatch"
            
            for q, s in zip(nl_list, sql_list):
                # Original sample processing
                s = s.strip()
                if s.endswith(";"):
                    s = s[:-1].strip()
                s = " ".join(s.split())
                
                # Original sample
                self._process_sample(q, s, tokenizer)
                
                # Data Augmentation
                augmentations = [
                    # Synonym replacements
                    q.replace('get', 'retrieve'),
                    q.replace('show', 'display'),
                    
                    # Structural variations
                    f"Give me {q}",
                    f"I want to know {q}",
                    
                    # Word order shuffle (with 50% probability)
                    " ".join(random.sample(q.split(), len(q.split()))) if random.random() < 0.5 else q
                ]
                
                # Process augmented samples
                for aug_q in augmentations:
                    self._process_sample(aug_q, s, tokenizer)
        
        elif split == "dev":
            nl_path  = os.path.join(data_folder, "dev.nl")
            sql_path = os.path.join(data_folder, "dev.sql")
            nl_list  = load_lines(nl_path)
            sql_list = load_lines(sql_path)
            assert len(nl_list) == len(sql_list), "dev NL/SQL size mismatch"
            
            for q, s in zip(nl_list, sql_list):
                s = s.strip()
                if s.endswith(";"):
                    s = s[:-1].strip()
                s = " ".join(s.split())
                
                src_text = "translate to SQL: " + q.strip()
                enc = tokenizer(
                    src_text,
                    padding=False,
                    truncation=True,
                    max_length=MAX_SRC_LEN,
                    return_tensors=None,
                )["input_ids"]
                
                dec = tokenizer(
                    s,
                    padding=False,
                    truncation=True,
                    max_length=MAX_TGT_LEN,
                    return_tensors=None,
                )["input_ids"]
                
                self.items.append({
                    "enc_ids": torch.tensor(enc, dtype=torch.long),
                    "dec_ids": torch.tensor(dec, dtype=torch.long),
                })
        
        elif split == "test":
            nl_path = os.path.join(data_folder, "test.nl")
            nl_list = load_lines(nl_path)
            
            for q in nl_list:
                src_text = "translate to SQL: " + q
                enc = tokenizer(
                    src_text, 
                    padding=False, 
                    truncation=True, 
                    max_length=MAX_SRC_LEN, 
                    return_tensors=None
                )["input_ids"]
                
                self.items.append({
                    "enc_ids": torch.tensor(enc, dtype=torch.long),
                    "dec_ids": None,  # no targets on test
                })
        
        else:
            raise ValueError(f"Unknown split: {split}")

    def _process_sample(self, query, sql, tokenizer):
        src_text = "translate to SQL: " + query.strip()
        
        enc = tokenizer(
            src_text,
            padding=False,
            truncation=True,
            max_length=MAX_SRC_LEN,
            return_tensors=None,
        )["input_ids"]
        
        dec = tokenizer(
            sql,
            padding=False,
            truncation=True,
            max_length=MAX_TGT_LEN,
            return_tensors=None,
        )["input_ids"]
        
        self.items.append({
            "enc_ids": torch.tensor(enc, dtype=torch.long),
            "dec_ids": torch.tensor(dec, dtype=torch.long),
        })


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def normal_collate_fn(batch):
    """
    Collate for train/dev:
      returns:
        encoder_ids (B,T)
        encoder_mask (B,T)  -- 1 for tokens, 0 for pad
        decoder_inputs (B,T')
        decoder_targets (B,T')
        initial_decoder_inputs (B,1)  -- first token BOS for generation
    """

    enc_list = [item["enc_ids"] for item in batch]
    encoder_ids = pad_sequence(enc_list, batch_first=True, padding_value=PAD_IDX)  # (B,T)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    # ---- Decoder ----
    dec_list = [item["dec_ids"] for item in batch]
    
    dec_in_list = []
    dec_tgt_list = []
    for dec in dec_list:
      
        if dec is None or len(dec) == 0:
            tgt = torch.tensor([PAD_IDX], dtype=torch.long)
        else:
            tgt = dec
        # Shift-right: [BOS] + tgt[:-1]
        if len(tgt) > 1:
            inp = torch.cat([torch.tensor([BOS_ID], dtype=torch.long), tgt[:-1]], dim=0)
        else:
            inp = torch.tensor([BOS_ID], dtype=torch.long)  # trivial case
        dec_in_list.append(inp)
        dec_tgt_list.append(tgt)

    decoder_inputs = pad_sequence(dec_in_list, batch_first=True, padding_value=PAD_IDX)   # (B,T')
    decoder_targets = pad_sequence(dec_tgt_list, batch_first=True, padding_value=PAD_IDX) # (B,T')


    initial_decoder_inputs = torch.full(
        (len(batch), 1), BOS_ID, dtype=torch.long
    )  # (B,1)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    """
    Collate for test:
      returns:
        encoder_ids (B,T)
        encoder_mask (B,T)
        initial_decoder_inputs (B,1)
    """
    enc_list = [item["enc_ids"] for item in batch]
    encoder_ids = pad_sequence(enc_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.full((len(batch), 1), BOS_ID, dtype=torch.long)
    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def load_prompting_data(data_folder):
    """
    Only needed for prompting baselines.
    Returns: train_x, train_y, dev_x, dev_y, test_x  (lists of strings)
    """
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x   = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y   = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x  = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x











'''def process_data(self, data_folder, split, tokenizer):
        if split == "train":
            nl_path  = os.path.join(data_folder, "train.nl")
            sql_path = os.path.join(data_folder, "train.sql")
            nl_list  = load_lines(nl_path)
            sql_list = load_lines(sql_path)
            assert len(nl_list) == len(sql_list), "train NL/SQL size mismatch"

            for q, s in zip(nl_list, sql_list):
                # --- normalize SQL a bit ---
                s = s.strip()
                if s.endswith(";"):
                    s = s[:-1].strip()         # drop trailing semicolons
                s = " ".join(s.split())        # collapse weird spaces/newlines

                # --- encoder input with task prefix ---
                src_text = "translate to SQL: " + q.strip()
                enc = tokenizer(
                    src_text,
                    padding=False,
                    truncation=True,
                    max_length=MAX_SRC_LEN,
                    return_tensors=None,
                )["input_ids"]

                # --- decoder target ---
                dec = tokenizer(
                    s,
                    padding=False,
                    truncation=True,
                    max_length=MAX_TGT_LEN,
                    return_tensors=None,
                )["input_ids"]

                self.items.append({
                    "enc_ids": torch.tensor(enc, dtype=torch.long),
                    "dec_ids": torch.tensor(dec, dtype=torch.long),
                })



        elif split == "dev":
            nl_path  = os.path.join(data_folder, "dev.nl")
            sql_path = os.path.join(data_folder, "dev.sql")
            nl_list  = load_lines(nl_path)
            sql_list = load_lines(sql_path)
            assert len(nl_list) == len(sql_list), "dev NL/SQL size mismatch"

            for q, s in zip(nl_list, sql_list):
                # --- normalize SQL a bit ---
                s = s.strip()
                if s.endswith(";"):
                    s = s[:-1].strip()         # drop trailing semicolons
                s = " ".join(s.split())        # collapse weird spaces/newlines

                # --- encoder input with task prefix ---
                src_text = "translate to SQL: " + q.strip()
                enc = tokenizer(
                    src_text,
                    padding=False,
                    truncation=True,
                    max_length=MAX_SRC_LEN,
                    return_tensors=None,
                )["input_ids"]

                # --- decoder target ---
                dec = tokenizer(
                    s,
                    padding=False,
                    truncation=True,
                    max_length=MAX_TGT_LEN,
                    return_tensors=None,
                )["input_ids"]

                self.items.append({
                    "enc_ids": torch.tensor(enc, dtype=torch.long),
                    "dec_ids": torch.tensor(dec, dtype=torch.long),
                })


        elif split == "test":
            nl_path = os.path.join(data_folder, "test.nl")
            nl_list = load_lines(nl_path)

            for q in nl_list:
                src_text = "translate to SQL: " + q
                enc = tokenizer(
                    src_text, padding=False, truncation=True, max_length=MAX_SRC_LEN, return_tensors=None
                )["input_ids"]

                self.items.append({
                    "enc_ids": torch.tensor(enc, dtype=torch.long),
                    "dec_ids": None,  # no targets on test
                })
        else:
            raise ValueError(f"Unknown split: {split}")'''