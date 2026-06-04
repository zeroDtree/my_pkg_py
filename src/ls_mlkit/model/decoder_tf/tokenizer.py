import collections
import copy
import json
import pickle

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class Tokenizer:
    def __init__(self) -> None:
        self.special_vocab: dict[str, int] | None = None
        self.inv_vocab: dict[int, str] | None = None
        self.vocab: dict[str, int] | None = None
        self.has_add_special_tokens = False
        self.has_build_vocab = False
        self.default_special_token_list = [
            "[BOS]",
            "[EOS]",
            "[UNK]",
            "[SEP]",
            "[PAD]",
            "[CLS]",
            "[MASK]",
        ]
        self.eos_token: str | None = None
        self.pad_token: str | None = None
        self.set_eos_token("[EOS]")
        self.set_pad_token("[PAD]")
        self.eos_id: int | None = None
        self.pad_id: int | None = None

    def set_eos_token(self, val: str = "[EOS]") -> None:
        self.eos_token = val

    def set_pad_token(self, val: str = "[PAD]") -> None:
        self.pad_token = val

    def tokenize(self, text: str) -> list[str]:
        return list(text)

    def convert_token_to_id(self, token_list: list[str]) -> list[int]:
        assert self.has_build_vocab, "haven't build vocab, please call <build_vocab> method fist! "
        assert self.vocab is not None
        id_list: list[int] = []
        for token in token_list:
            id_list.append(self.vocab.get(token, self.vocab.get("UNK", 0)))
        return id_list

    def convert_id_to_token(self, id_list: list[int]) -> list[str]:
        assert self.inv_vocab is not None
        token_list: list[str] = []
        for index in id_list:
            token_list.append(self.inv_vocab.get(index, "[UNK]"))
        return token_list

    def build_vocab(self, text_list: list[str], max_vocab_size: int = 10000, min_freq: int = 1) -> None:
        counter = collections.Counter()
        p_bar = tqdm(total=len(text_list), desc="counting token in texts")
        for text in text_list:
            tokens = list(text)
            counter.update(tokens)
            p_bar.update(1)

        if not self.has_add_special_tokens:
            self.add_special_tokens(self.default_special_token_list)
        assert self.special_vocab is not None
        vocab: dict[str, int] = copy.deepcopy(self.special_vocab)
        p_bar = tqdm(
            total=max(counter.total(), max_vocab_size),
            desc="specifying id to tokens",
        )
        for token, freq in counter.most_common(max_vocab_size):
            if freq >= min_freq and token not in vocab:
                vocab[token] = len(vocab)
            p_bar.update(1)

        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.has_build_vocab = True
        assert self.eos_token is not None and self.pad_token is not None
        self.eos_id = vocab[self.eos_token]
        self.pad_id = vocab[self.pad_token]

    def add_special_tokens(self, special_token_list: list[str]) -> None:
        self.special_vocab = {special_token_list[i]: i for i in range(len(special_token_list))}
        self.has_add_special_tokens = True

    def save_state_dict(self, save_directory: str = "model_pretrained/gpt2") -> None:
        assert self.vocab is not None
        state_dict = self.__dict__
        state_dict["vocab"] = self.vocab
        with open(f"{save_directory}/tokenizer.pkl", "wb") as f:
            pickle.dump(self.__dict__, f)
        with open(f"{save_directory}/vocab.json", "w") as f:
            json.dump(self.vocab, f)

    def load_state_dict(self, save_directory: str = "model_pretrained/gpt2") -> None:
        with open(f"{save_directory}/tokenizer.pkl", "rb") as f:
            state_dict = pickle.load(f)
            self.__dict__.update(state_dict)

    def get_vocab_size(self) -> int:
        assert self.vocab is not None
        return len(self.vocab)


def get_masks(data: torch.Tensor, tokenizer: Tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = data.shape[1]
    attention_mask = (
        (torch.ones(seq_len, seq_len) - torch.triu(torch.ones(seq_len, seq_len))).type(torch.bool).transpose(0, 1)
    )
    assert tokenizer.pad_id is not None
    padding_mask = data == tokenizer.pad_id
    return attention_mask, padding_mask


def get_collate_fn(tokenizer: Tokenizer, max_len: int = 500, train: bool = True):
    def transform_text_to_tensor(text: str, tokenizer: Tokenizer) -> torch.Tensor:
        assert tokenizer.eos_token is not None
        return torch.Tensor(
            tokenizer.convert_token_to_id(tokenizer.tokenize(text) + ([tokenizer.eos_token] if train else []))
        )

    def collate_fn(batch: list[str]) -> dict[str, torch.Tensor]:
        collated_batch: list[torch.Tensor] = []
        for sample in batch:
            collated_batch.append(transform_text_to_tensor(sample.rstrip("\n"), tokenizer))
        assert tokenizer.pad_token is not None
        padded_batch = pad_sequence(
            collated_batch,
            padding_value=tokenizer.convert_token_to_id([tokenizer.pad_token])[0],
            batch_first=True,
        )
        collated_batch_tensor = padded_batch.long()[:, :max_len]
        attention_mask, padding_mask = get_masks(
            collated_batch_tensor[:, :-1] if train else collated_batch_tensor,
            tokenizer,
        )
        result = {
            "x": collated_batch_tensor[:, :-1] if train else collated_batch_tensor,
            "y": collated_batch_tensor[:, 1:],
            "att_mask": attention_mask,
            "pad_mask": padding_mask,
        }
        return result

    return collate_fn
