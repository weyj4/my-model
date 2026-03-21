from datasets import load_dataset
import tiktoken
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
    def __init__(self, tokens: list[int], context_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []
        for i in range(0, len(tokens) - context_length, stride):
            self.input_ids.append(torch.tensor(tokens[i:i+context_length]))
            self.target_ids.append(torch.tensor(tokens[i+1:i+context_length+1]))
    
    def __len__(self): return len(self.input_ids)
    def __getitem__(self, idx): return self.input_ids[idx], self.target_ids[idx]

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

def create_fineweb_loaders(
    tokenizer, 
    num_tokens: int,
    batch_size: int,
    context_length: int,
    val_ratio: float = 0.05,
):
    
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True
    )
    
    tokens = []
    for example in dataset:
        tokens.extend(tokenizer.encode(example["text"], allowed_special={"<|endoftext|>"}))
        tokens.append(tokenizer.eot_token)
        if len(tokens) >= num_tokens:
            break
    tokens = tokens[:num_tokens]
    
    split = int(len(tokens) * (1 - val_ratio))
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]
    
    # convert to text for GPTDatasetV1
    # actually cleaner to make a TokenDataset that takes raw tokens:
    train_loader = create_token_dataloader(
        train_tokens, batch_size, context_length, shuffle=True
    )
    val_loader = create_token_dataloader(
        val_tokens, batch_size, context_length, shuffle=False
    )
    return train_loader, val_loader
