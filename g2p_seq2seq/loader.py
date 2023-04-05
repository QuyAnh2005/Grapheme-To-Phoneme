import torch

from torch.utils.data import Dataset, DataLoader
from .utils import Vocab, Tokenize, read_file


# Set up the data loader
class GraphemePhonemeDataset(Dataset):
    def __init__(
            self,
            data,
            grapheme_tokenizer,
            phoneme_tokenizer,
            max_length_grapheme,
            max_length_phoneme
    ):

        self.data = data
        self.grapheme_tokenizer = grapheme_tokenizer
        self.phoneme_tokenizer = phoneme_tokenizer
        self.max_length_grapheme = max_length_grapheme
        self.max_length_phoneme = max_length_phoneme

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        grapheme, phoneme = self.data[idx]

        tokenized_g = self.grapheme_tokenizer.tokenize(list(grapheme), self.max_length_grapheme)
        tokenized_p = self.phoneme_tokenizer.tokenize(phoneme, self.max_length_phoneme)

        return tokenized_g, tokenized_p
