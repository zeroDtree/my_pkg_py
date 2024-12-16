from torch.utils.data import Dataset, random_split
import random
import torch


class CyclicGroupGenerator:
    def __init__(self, a: int, p: int):
        self.a = a
        self.p = p
        self.x = 0

    def next(self):
        self.x = (self.a + self.x) % self.p
        return self.x


class LDADataset(Dataset):
    """
    Dataset for LDA. no need to be tokenized.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_local_topics: int = 1,
        n_total_topics: int = 10,
        n_words_per_topic: int = 7,
        seq_len: int = 100,
        fix_seq_len: bool = True,
        seed: int = 31,
        local_topic_strategy: str = "cyclic",
        fix_local_topics_num: bool = True,
    ):
        self.n_samples = n_samples
        self.n_local_topics = n_local_topics
        self.n_total_topics = n_total_topics
        self.n_words_per_topic = n_words_per_topic
        self.seq_len = seq_len
        self.fix_seq_len = fix_seq_len
        self.seed = seed
        self.local_topic_strategy = local_topic_strategy
        data = LDADataset.generate_dataset(
            n_samples=n_samples,
            n_local_topics=n_local_topics,
            n_total_topics=n_total_topics,
            n_words_per_topic=n_words_per_topic,
            seq_len=seq_len,
            fix_seq_len=fix_seq_len,
            seed=seed,
            local_topic_strategy=local_topic_strategy,
            fix_local_topics_num=fix_local_topics_num,
        )
        self.data = self.data_list_to_tensor(data)

    def generate_dataset(
        n_samples: int,
        n_local_topics: int,
        n_total_topics: int,
        n_words_per_topic: int,
        seq_len: int,
        fix_local_topics_num: bool = True,
        fix_seq_len: bool = True,
        seed: int = 31,
        local_topic_strategy: str = "cyclic",
    ):
        random.seed(seed)
        data_list = []
        number_generator_list = []
        for idx_topic in range(n_total_topics):
            p = n_words_per_topic
            a = random.randint(1, p)
            if local_topic_strategy == "cyclic":
                number_generator_list.append(CyclicGroupGenerator(a=a, p=p))
            else:
                raise ValueError(
                    f"Invalid local topic strategy: {local_topic_strategy}"
                )
        for idx_sample in range(n_samples):
            sample = []
            local_topic_list = []
            if not fix_local_topics_num:
                n_local_topics = random.randint(1, n_total_topics)
            for idx_topic in range(n_local_topics):
                local_topic_list.append(random.randint(0, n_total_topics - 1))
            if not fix_seq_len:
                seq_len = random.randint(1, seq_len)
            for idx_word in range(seq_len):
                topic = random.choice(local_topic_list)
                word = number_generator_list[topic].next() + topic * n_words_per_topic
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
    n_samples: int = 100,
    n_local_topics: int = 1,
    n_total_topics: int = 10,
    n_words_per_topic: int = 7,
    seq_len: int = 100,
    eval_ratio: float = 0.1,
    fix_seq_len: bool = True,
    fix_local_topics_num: bool = True,
    local_topic_strategy: str = "cyclic",
):
    dataset = LDADataset(
        n_samples=n_samples,
        n_local_topics=n_local_topics,
        n_total_topics=n_total_topics,
        n_words_per_topic=n_words_per_topic,
        seq_len=seq_len,
        fix_seq_len=fix_seq_len,
        seed=seed,
        fix_local_topics_num=fix_local_topics_num,
        local_topic_strategy=local_topic_strategy,
    )
    dataset_train, dataset_eval = random_split(dataset, [1 - eval_ratio, eval_ratio])
    return dataset_train, dataset_eval, dataset_eval


if __name__ == "__main__":
    seed = random.randint(0, 1000000)
    dataset_train, dataset_eval, dataset_test = get_lda_dataset(
        seed=seed,
        n_samples=20,
        n_local_topics=1,
        n_total_topics=10,
        n_words_per_topic=7,
        seq_len=20,
    )
    print(dataset_train[0])
