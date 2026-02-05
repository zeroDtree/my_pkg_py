import random

import datasets
from tqdm import tqdm

from ..util.decorators import cache_to_disk


def _int32(x):
    return int(0xFFFFFFFF & x)


class MT19937:
    def __init__(self, seed):
        self.mt = [0] * 624
        self.mt[0] = seed
        self.mti = 0
        for i in range(1, 624):
            self.mt[i] = _int32(1812433253 * (self.mt[i - 1] ^ self.mt[i - 1] >> 30) + i)

    def extract_number(self):
        if self.mti == 0:
            self.twist()
        y = self.mt[self.mti]
        y = y ^ y >> 11
        y = y ^ y << 7 & 2636928640
        y = y ^ y << 15 & 4022730752
        y = y ^ y >> 18
        self.mti = (self.mti + 1) % 624
        return _int32(y)

    def twist(self):
        for i in range(0, 624):
            y = _int32((self.mt[i] & 0x80000000) + (self.mt[(i + 1) % 624] & 0x7FFFFFFF))
            self.mt[i] = (y >> 1) ^ self.mt[(i + 397) % 624]

            if y % 2 != 0:
                self.mt[i] = self.mt[i] ^ 0x9908B0DF


def _int8(x):
    return int(0xFF & x)


class MT8Bit(MT19937):
    def extract_number(self):
        return _int8(super().extract_number())


def _int12(x):
    return int(0xFFF & x)


class MT12Bit(MT19937):
    def extract_number(self):
        return _int12(super().extract_number())


def _int16(x):
    return int(0xFFFF & x)


class MT16Bit(MT19937):
    def extract_number(self):
        return _int16(super().extract_number())


def _int2(x):
    return int(0x1 & x)


class MT2Bit(MT19937):
    def extract_number(self):
        return _int2(super().extract_number())


@cache_to_disk()
def load_mt19937_binary_stream(
    max_seq_len=666,
    num_samples=10000,
    seed=31,
    eval_split_ratio=0.1,
    fixed_len=True,
    min_seq_len=7,
):
    random.seed(seed)
    train_size = num_samples - eval_split_ratio * num_samples
    sample_list = []
    for i in tqdm(range(num_samples)):
        len = max_seq_len if fixed_len else random.randint(min_seq_len, max_seq_len)
        seed = random.randint(0, int(1e9))
        mt = MT2Bit(seed)
        seq = ""
        for j in range(len):
            random_number = mt.extract_number()
            seq += str(random_number)
        sample_list.append({"seed": str(_int2(seed)), "seq": seq})

    def preprocess(data):
        return {
            "x": data["seed"],
            "y": data["seq"],
        }

    train_samples = []
    eval_samples = []
    count = 0
    bar = tqdm(total=num_samples)
    total = 0
    ok = 0
    for sample in sample_list:
        total += 1
        ok += 1
        bar.set_description(f"ok: {ok}/{total}")
        bar.update(1)
        processed_sample = preprocess(sample)
        if count < train_size:
            train_samples.append(processed_sample)
        elif train_size <= count < num_samples:
            eval_samples.append(processed_sample)
        elif count >= num_samples:
            break
        count += 1
    train_set = datasets.Dataset.from_list(train_samples)
    eval_set = datasets.Dataset.from_list(eval_samples)
    return train_set, eval_set, eval_set


@cache_to_disk()
def load_mt19937(
    max_seq_len=666,  # the number of random numbers in a sequence
    num_samples=100000,
    eval_split_ratio=0.1,
    seed=31,
    fixed_len=False,
    delimiter=",",
    num_bits=8,
):
    random.seed(seed)
    # total 395000 samples
    train_size = num_samples - eval_split_ratio * num_samples
    sample_list = []
    for i in tqdm(range(num_samples)):
        len = max_seq_len if fixed_len else random.randint(7, max_seq_len)
        seed = random.randint(0, int(1e9))
        if num_bits == 8:
            seed = _int8(seed)
        elif num_bits == 32:
            seed = _int32(seed)
        elif num_bits == 12:
            seed = _int12(seed)
        elif num_bits == 16:
            seed = _int16(seed)
        mt = None
        if num_bits == 32:
            mt = MT19937(seed)
        elif num_bits == 8:
            mt = MT8Bit(seed)
        elif num_bits == 12:
            mt = MT12Bit(seed)
        elif num_bits == 16:
            mt = MT16Bit(seed)
        seq = ""
        for j in range(len):
            random_number = mt.extract_number()
            seq += delimiter + str(random_number)
        sample_list.append({"seed": str(seed), "seq": seq})

    def preprocess(data):
        return {
            "x": data["seed"],
            "y": data["seq"],
        }

    train_samples = []
    eval_samples = []
    count = 0
    bar = tqdm(total=num_samples)
    total = 0
    ok = 0
    for sample in sample_list:
        total += 1
        ok += 1
        bar.set_description(f"ok: {ok}/{total}")
        bar.update(1)
        processed_sample = preprocess(sample)
        if count < train_size:
            train_samples.append(processed_sample)
        elif train_size <= count < num_samples:
            eval_samples.append(processed_sample)
        elif count >= num_samples:
            break
        count += 1
    train_set = datasets.Dataset.from_list(train_samples)
    eval_set = datasets.Dataset.from_list(eval_samples)
    return train_set, eval_set, eval_set


def load_mt19937_8bits(seed=31, **kwargs):
    train_set, validation_set, test_set = load_mt19937(
        max_seq_len=256,
        num_samples=256,
        eval_split_ratio=0.0,
        seed=seed,
        delimiter=",",
        fixed_len=True,
        num_bits=8,
    )
    return train_set, train_set, train_set


def load_mt19937_8bits_with_eval(seed=31, **kwargs):
    train_set, validation_set, test_set = load_mt19937(
        max_seq_len=256,
        num_samples=256,
        eval_split_ratio=0.1,
        seed=seed,
        delimiter=",",
        fixed_len=True,
        num_bits=8,
    )
    return train_set, validation_set, test_set


def load_mt19937_12bits(seed=31, **kwargs):
    train_set, validation_set, test_set = load_mt19937(
        max_seq_len=4096,
        num_samples=4096,
        eval_split_ratio=0.0,
        seed=seed,
        delimiter=",",
        fixed_len=True,
        num_bits=12,
    )
    return train_set, train_set, train_set


def load_mt19937_12bits_with_eval(seed=31, **kwargs):
    train_set, validation_set, test_set = load_mt19937(
        max_seq_len=4096,
        num_samples=4096,
        eval_split_ratio=0.05,
        seed=seed,
        delimiter=",",
        fixed_len=True,
        num_bits=12,
    )
    return train_set, validation_set, test_set


def load_mt19937_16bits(seed=31, **kwargs):
    train_set, validation_set, test_set = load_mt19937(
        max_seq_len=4096,
        num_samples=4096,
        eval_split_ratio=0.0,
        seed=seed,
        delimiter=",",
        fixed_len=True,
        num_bits=16,
    )
    return train_set, train_set, train_set


def load_mt19937_16bits_with_eval(seed=31, **kwargs):
    train_set, validation_set, test_set = load_mt19937(
        max_seq_len=4096,
        num_samples=4096,
        eval_split_ratio=0.05,
        seed=seed,
        delimiter=",",
        fixed_len=True,
        num_bits=16,
    )
    return train_set, validation_set, test_set


def load_mt19937_32bits(seed=31, **kwargs):
    train_set, validation_set, test_set = load_mt19937(
        max_seq_len=1024,
        num_samples=100000,
        eval_split_ratio=0,
        seed=seed,
        delimiter=",",
        num_bits=32,
    )
    return train_set, train_set, train_set


def load_mt19937_32bits_with_eval(seed=31, **kwargs):
    train_set, validation_set, test_set = load_mt19937(
        max_seq_len=1024,
        num_samples=100000,
        eval_split_ratio=0.01,
        seed=seed,
        delimiter=",",
        num_bits=32,
    )
    return train_set, validation_set, test_set


if __name__ == "__main__":
    pass
