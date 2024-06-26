import torch
from torch.utils.data import Dataset
from typing import List, Tuple

class ShakespeareDataset(Dataset):
    def __init__(self, text: str, sequence_length: int, tokenizer):
        self.text = text
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.data = self.tokenizer.encode(self.text)

    def __len__(self) -> int:
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq = self.data[idx:idx + self.sequence_length]
        target_seq = self.data[idx + 1:idx + self.sequence_length + 1]
        return torch.tensor(input_seq), torch.tensor(target_seq)