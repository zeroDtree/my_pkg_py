from torch.utils.data import Dataset, random_split
import random
import torch


class LDADataset(Dataset):
    """
    Dataset for LDA. no need to be tokenized.
    """

    def __init__(
        self,
        n_samples: int = 200,
        n_local_topics: int = 3,
        n_total_topics: int = 10,
        n_words_per_topic: int = 10,
        seq_len: int = 10,
        fix_seq_len: bool = True,
        seed: int = 31,
    ):
        self.n_samples = n_samples
        self.n_local_topics = n_local_topics
        self.n_total_topics = n_total_topics
        self.n_words_per_topic = n_words_per_topic
        self.seq_len = seq_len
        self.fix_seq_len = fix_seq_len
        self.seed = seed
        data = LDADataset.generate_dataset(
            n_samples=n_samples,
            n_local_topics=n_local_topics,
            n_total_topics=n_total_topics,
            n_words_per_topic=n_words_per_topic,
            seq_len=seq_len,
            fix_seq_len=fix_seq_len,
            seed=seed,
        )
        self.data = self.data_list_to_tensor(data)

    def generate_dataset(
        n_samples: int,
        n_local_topics: int,
        n_total_topics: int,
        n_words_per_topic: int,
        seq_len: int,
        fix_seq_len: bool = True,
        seed: int = 31,
    ):
        random.seed(seed)
        data_list = []
        for idx_sample in range(n_samples):
            sample = []
            local_topic_list = []
            for idx_topic in range(n_local_topics):
                local_topic_list.append(random.randint(1, n_total_topics))
            if not fix_seq_len:
                seq_len = random.randint(1, seq_len)
            for idx_word in range(seq_len):
                topic = random.choice(local_topic_list)
                word = random.randint(
                    (topic - 1) * n_words_per_topic + 1, topic * n_words_per_topic
                )
                sample.append(word)
            data_list.append(sample)
        return data_list

    def data_list_to_tensor(self, data_list: list[list[int]]):
        return torch.tensor(data_list, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_lda_dataset(
    seed: int = 31,
    n_samples: int = 10000,
    n_local_topics: int = 2,
    n_total_topics: int = 10,
    n_words_per_topic: int = 10,
    seq_len: int = 10,
    fix_seq_len: bool = True,
    eval_ratio: float = 0.1,
):
    dataset = LDADataset(
        n_samples=n_samples,
        n_local_topics=n_local_topics,
        n_total_topics=n_total_topics,
        n_words_per_topic=n_words_per_topic,
        seq_len=seq_len,
        fix_seq_len=fix_seq_len,
        seed=seed,
    )
    dataset_train, dataset_eval = random_split(dataset, [1 - eval_ratio, eval_ratio])
    return dataset_train, dataset_eval, dataset_eval
