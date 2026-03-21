from datasets import load_dataset
import tiktoken
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

class TokenDataset(Dataset):
    def __init__(self, tokens: list[int] | np.ndarray, context_length: int, stride: int):
        self.context_length = context_length
        self.stride = stride

        if isinstance(tokens, np.ndarray):
            self.tokens = tokens
            self.use_numpy = True
            self.length = (len(tokens) - context_length) // stride
        else:
            self.use_numpy = False
            self.input_ids = []
            self.target_ids = []
            for i in range(0, len(tokens) - context_length, stride):
                self.input_ids.append(torch.tensor(tokens[i:i+context_length]))
                self.target_ids.append(torch.tensor(tokens[i+1:i+context_length+1]))
    
    def __len__(self):
        if self.use_numpy:
            return self.length
        return len(self.input_ids)

    def __getitem__(self, idx):
        if self.use_numpy:
            start = idx * self.stride
            x = torch.from_numpy(
                self.tokens[start:start + self.context_length].astype(np.int64)
            )
            y = torch.from_numpy(
                self.tokens[start + 1:start + self.context_length + 1].astype(np.int64)
            )
            return x, y
        return self.input_ids[idx], self.target_ids[idx]

def create_token_dataloader(
    tokens: list[int],
    batch_size: int,
    context_length: int,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = TokenDataset(tokens, context_length, stride=context_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # faster CPU->GPU transfer
    )

def create_verdict_loaders(
    batch_size: int,
    context_length: int,
    val_ratio: float = 0.1,
) -> tuple[DataLoader, DataLoader]:
    """This basically just tokenizes the raw text and creates
    a train/val split"""
    tokenizer = tiktoken.get_encoding("gpt2")
    
    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    
    split = int(len(tokens) * (1 - val_ratio))
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]
    
    train_loader = create_token_dataloader(
        train_tokens, batch_size, context_length, shuffle=True
    )
    val_loader = create_token_dataloader(
        val_tokens, batch_size, context_length, shuffle=False, drop_last=False
    )
    return train_loader, val_loader

def create_fineweb_loaders_from_file(
    path: str,
    batch_size: int,
    context_length: int,
    val_ratio: float = 0.05,
) -> tuple[DataLoader, DataLoader]:
    tokens = np.load(path, mmap_mode='r')  # memory-mapped — doesn't load into RAM
    split = int(len(tokens) * (1 - val_ratio))
    
    train_loader = create_token_dataloader(
        tokens[:split], batch_size, context_length, shuffle=True
    )
    val_loader = create_token_dataloader(
        tokens[split:], batch_size, context_length, shuffle=False, drop_last=False
    )
    return train_loader, val_loader
