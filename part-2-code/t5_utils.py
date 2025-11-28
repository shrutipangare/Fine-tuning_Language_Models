import os
import torch
import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
    wandb.login(key=os.environ.get("WANDB_API_KEY", None))
    wandb.init(project="t5-text2sql", name=args.experiment_name, config=vars(args))

# def initialize_model(args):
#     '''
#     Helper function to initialize the model. You should be either finetuning
#     the pretrained model associated with the 'google-t5/t5-small' checkpoint
#     or training a T5 model initialized with the 'google-t5/t5-small' config
#     from scratch.
#     '''
#     pass

from load_data import TOKENIZER, PAD_IDX  # add this import at the top of t5_utils.py

'''def initialize_model(args):
    """
    Initialize T5 from scratch:
    - Truly random initialization of weights
    - Configure tokenizer alignment
    - Setup model for training from scratch
    """
    # Use the T5 config but initialize from scratch
    cfg = T5Config.from_pretrained("google-t5/t5-small")
    model = T5ForConditionalGeneration(cfg)

    # Resize token embeddings to match tokenizer
    model.resize_token_embeddings(len(TOKENIZER))

    # Ensure model config matches tokenizer
    model.config.pad_token_id = PAD_IDX
    model.config.decoder_start_token_id = PAD_IDX
    model.config.eos_token_id = TOKENIZER.eos_token_id

    # Move model to device
    model.to(DEVICE)
    return model'''
def initialize_model(args): #used for fintuning
    """
    Initialize T5:
      - finetune=True  -> load pretrained "google-t5/t5-small"
      - finetune=False -> initialize from its config (scratch)
    Also:
      - ensure pad / decoder_start / eos ids match our tokenizer
      - (optionally) freeze some encoder layers for stability / ablations
    """
    model_name = "google-t5/t5-small"

    if args.finetune:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        cfg = T5Config.from_pretrained(model_name)
        model = T5ForConditionalGeneration(cfg)

    # Make sure model & tokenizer are fully aligned
    model.resize_token_embeddings(len(TOKENIZER))

    model.config.pad_token_id = PAD_IDX
    model.config.decoder_start_token_id = PAD_IDX   # we use pad as BOS in load_data.py
    if model.config.eos_token_id is None:
        model.config.eos_token_id = TOKENIZER.eos_token_id

    # OPTIONAL: freeze some bottom encoder layers (helps stability, and is an "architecture choice")
    # You can tune this number or add an arg; here we freeze the first 2 encoder blocks by default.
    num_freeze = getattr(args, "freeze_encoder_layers", 0)  # if you add this arg in train_t5.py
    if num_freeze > 0:
        freeze_encoder_bottom_k_layers(model, k=num_freeze)

    model.to(DEVICE)
    return model


def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    # Save model checkpoint to be able to load the model later
    sub = "best" if best else "last"
    outdir = os.path.join(checkpoint_dir, sub)
    mkdir(outdir)
    model.save_pretrained(outdir)

def load_model_from_checkpoint(args, best):
    '''
    Load a T5 model from a checkpoint saved by save_model().
    '''
    sub = "best" if best else "last"
    ckpt_dir = os.path.join(args.checkpoint_dir, sub)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint folder not found: {ckpt_dir}")
    model = T5ForConditionalGeneration.from_pretrained(ckpt_dir).to(DEVICE)
    model.eval()
    return model

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    result += list(model._parameters.keys())
    return result



def freeze_encoder_bottom_k_layers(model, k=2):
    """
    Freeze the bottom-k encoder layers of T5.
    This is a common fine-tuning trick: keep low-level representations fixed,
    only adapt higher-level + decoder.
    """
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "block"):
        return

    encoder_blocks = model.encoder.block
    k = min(k, len(encoder_blocks))

    for i in range(k):
        for param in encoder_blocks[i].parameters():
            param.requires_grad = False

