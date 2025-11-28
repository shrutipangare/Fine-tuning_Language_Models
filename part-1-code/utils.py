import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.
def get_wordnet_pos(treebank_tag):
    """Convert between a treebank tag to a wordnet POS tag"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.
def get_synonym(word, pos):
    """Find a synonym for a given word"""
    synonyms = []
    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            # Avoid adding the original word and prefer single-word synonyms
            if lemma.name().lower() != word.lower() and '_' not in lemma.name():
                synonyms.append(lemma.name())
        if synonyms:
            return random.choice(synonyms)
    return word


    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    # Tokenize and tag the words

def custom_transform(example):
    words = nltk.word_tokenize(example["text"].lower())
    tagged_words = nltk.pos_tag(words)
    transformed_words = []
    
    for word, tag in tagged_words:
        # More diverse transformation strategies
        if random.random() < 0.5:
            wordnet_pos = get_wordnet_pos(tag)
            if wordnet_pos:
                synonym = get_synonym(word, wordnet_pos)
                transformed_words.append(synonym)
            else:
                transformed_words.append(word)
        else:
            # Keep original word
            transformed_words.append(word)
    
    transformed_text = TreebankWordDetokenizer().detokenize(transformed_words)
    example["text"] = transformed_text
    return example    
