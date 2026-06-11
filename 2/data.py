import json
from datasets import load_dataset, concatenate_datasets
import tiktoken
import os
import numpy as np
import torch

def create_tokenizer():
    base_tokenizer = tiktoken.get_encoding("cl100k_base")
    
    custom_tokens = ["<|startoftext|>", "<|endoftext|>"]
    custom_token_ids = {
        token: base_tokenizer.n_vocab + i for i, token in enumerate(custom_tokens)
    }
    
    extended_tokenizer = tiktoken.Encoding(
        name="custom_cl100k",
        pat_str=base_tokenizer._pat_str,
        mergeable_ranks=base_tokenizer._mergeable_ranks,
        special_tokens={**base_tokenizer._special_tokens, **custom_token_ids},
    )
    return extended_tokenizer

def _format_data(dataset, tokenizer):
    start = tokenizer.encode("<|startoftext|>", allowed_special="all")
    end = tokenizer.encode("<|endoftext|>", allowed_special="all")
    data = []
    for item in dataset:
        data.extend(start)
        data.extend(tokenizer.encode(item['text'], allowed_special="all"))
        data.extend(end)
    return np.array(data, dtype=np.uint32)

def download_data(tokenizer):
    os.makedirs('1/.cache', exist_ok=True)
    
    if not os.path.isfile('1/.cache/train.bin'):
        print("Downloading + tokenizing train split...")
        # take the datasets 0-19
        dataset1 = load_dataset(
            "mlfoundations/dclm-baseline-1.0", 
            data_files="global-shard_01_of_10/local-shard_0_of_10/shard_0000000*_processed.jsonl.zst", 
            split="train",
            streaming=True
        )
        dataset2 = load_dataset(
            "mlfoundations/dclm-baseline-1.0", 
            data_files="global-shard_01_of_10/local-shard_0_of_10/shard_0000001*_processed.jsonl.zst", 
            split="train",
            streaming=True
        )
        dataset = concatenate_datasets([dataset1, dataset2])
        data = _format_data(dataset, tokenizer)
        data.tofile('1/.cache/train.bin')
        print(f"Train tokens: {len(data):,}")

    if not os.path.isfile('1/.cache/val.bin'):
        print("Downloading & tokenizing val split...")
        dataset = load_dataset(
            "mlfoundations/dclm-baseline-1.0", 
            data_files="global-shard_01_of_10/local-shard_0_of_10/shard_00000020_processed.jsonl.zst", 
            split="train",
            streaming=True
        )
        data = _format_data(dataset, tokenizer)
        data.tofile('1/.cache/val.bin')
        print(f"Val tokens: {len(data):,}")


def get_batch(split, block_size, device, batch_size):
    path = '1/.cache/train.bin' if split == 'train' else '1/.cache/val.bin'
    data = np.memmap(path, dtype=np.uint32, mode='r')
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    
    if torch.cuda.is_available():
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"device : {device}")

block_size = 512
batch_size = 16 * 4

print("LOG: Create tokenizer")
tokenizer = create_tokenizer()
vocab_size = tiktoken.get_encoding("cl100k_base").n_vocab + 2

print("LOG: Download data")
download_data(tokenizer)



