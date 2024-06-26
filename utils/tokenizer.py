from typing import List

class CharacterTokenizer:
    def __init__(self, text: str):
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(set(text)))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx[char] for char in text]

    def decode(self, indices: List[int]) -> str:
        return ''.join(self.idx_to_char[idx] for idx in indices)