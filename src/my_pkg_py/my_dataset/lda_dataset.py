from torch.utils.data import Dataset, random_split
import random
import torch
from my_utils import seed_everything
from typing import Literal
import numpy as np
import pandas as pd


def softmax(x):
    x -= np.max(x, axis=-1, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    return x


class RandomGenerator:
    def __init__(
        self,
    ):
        pass

    def get_probabilities_by_distribution(
        n_numbers: int,
        number_list: list[int],
        distribution: Literal["uniform", "normal", "binomial", "linear"] = "uniform",
    ):
        if distribution == "uniform":
            p = np.ones(len(number_list)) / len(number_list)
        elif distribution == "normal":
            p = np.random.normal(loc=0, scale=1, size=len(number_list))
            p = softmax(p)
        elif distribution == "binomial":
            p = np.random.binomial(size=len(number_list), n=10, p=0.5)
            p = softmax(p)
        elif distribution == "linear":
            p = np.linspace(0, 1, len(number_list))
            p = softmax(p)
        else:
            raise ValueError(f"Invalid distribution: {distribution}")

        return p


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
        per_topic_strategy: str = "cyclic",
        fix_local_topics_num: bool = True,
        topic_distribution: str = "uniform",
    ):
        self.n_samples = n_samples
        self.n_local_topics = n_local_topics
        self.n_total_topics = n_total_topics
        self.n_words_per_topic = n_words_per_topic
        self.seq_len = seq_len
        self.fix_seq_len = fix_seq_len
        self.seed = seed
        self.per_topic_strategy = per_topic_strategy
        data = LDADataset.generate_dataset(
            n_samples=n_samples,
            n_local_topics=n_local_topics,
            n_total_topics=n_total_topics,
            n_words_per_topic=n_words_per_topic,
            seq_len=seq_len,
            fix_seq_len=fix_seq_len,
            seed=seed,
            per_topic_strategy=per_topic_strategy,
            fix_local_topics_num=fix_local_topics_num,
            topic_distribution=topic_distribution,
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
        per_topic_strategy: str = "cyclic",
        topic_distribution: str = "uniform",
    ):
        seed_everything(seed)
        p_topic_list = RandomGenerator.get_probabilities_by_distribution(
            n_numbers=n_total_topics,
            number_list=list(range(n_total_topics)),
            distribution=topic_distribution,
        )
        number_generator_list = []
        for _ in range(n_total_topics):
            p = n_words_per_topic
            a = random.randint(1, p)
            if per_topic_strategy == "cyclic":
                number_generator_list.append(CyclicGroupGenerator(a=a, p=p))
            else:
                raise ValueError(f"Invalid local topic strategy: {per_topic_strategy}")
        data_list = []
        for _ in range(n_samples):
            sample = []
            if not fix_local_topics_num:
                n_local_topics = random.randint(1, n_total_topics)
            local_topic_array = np.random.choice(
                a=range(n_total_topics), p=p_topic_list, size=n_local_topics
            )
            p_local_topic_array = p_topic_list[local_topic_array]
            print(
                f"p_local_topic_array: {p_local_topic_array}, p={p_local_topic_array / p_local_topic_array.sum()}"
            )
            p_local_topic_array /= p_local_topic_array.sum()

            if not fix_seq_len:
                seq_len = random.randint(1, seq_len)
            for idx_word in range(seq_len):
                topic = np.random.choice(a=local_topic_array, p=p_local_topic_array)
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
    per_topic_strategy: str = "cyclic",
    topic_distribution: str = "uniform",
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
        per_topic_strategy=per_topic_strategy,
        topic_distribution=topic_distribution,
    )
    dataset_train, dataset_eval = random_split(dataset, [1 - eval_ratio, eval_ratio])
    return dataset_train, dataset_eval, dataset_eval


# if __name__ == "__main__":
#     # Test RandomGenerator
#     import matplotlib.pyplot as plt

#     # Test different distributions
#     test_list = list(range(10))
#     n_samples = 10000
#     distributions = ["uniform", "normal", "binomial", "linear"]

#     fig, axes = plt.subplots(2, 2, figsize=(12, 8))
#     axes = axes.ravel()

#     for i, dist in enumerate(distributions):
#         numbers = RandomGenerator.get_numbers_from_a_list_by_distribution(
#             n_numbers=n_samples,
#             number_list=test_list,
#             distribution=dist
#         )

#         # Plot histogram
#         axes[i].hist(numbers, bins=len(test_list), density=True, alpha=0.7, color="skyblue")
#         axes[i].set_title(f"{dist} Distribution")
#         axes[i].set_xlabel("Value")
#         axes[i].set_ylabel("Frequency")
#         axes[i].grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.show()


if __name__ == "__main__":
    # Test different local_topic_distribution
    import matplotlib.pyplot as plt

    seed = 42
    n_samples = 1000
    n_local_topics = 2
    n_total_topics = 10
    n_words_per_topic = 7
    seq_len = 100

    distributions = ["uniform", "normal", "binomial", "linear"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    results = {"topic": list(range(n_total_topics))}

    for i, dist in enumerate(distributions):
        results[dist] = []
        dataset_train, _, _ = get_lda_dataset(
            seed=seed,
            n_samples=n_samples,
            n_local_topics=n_local_topics,
            n_total_topics=n_total_topics,
            n_words_per_topic=n_words_per_topic,
            seq_len=seq_len,
            topic_distribution=dist,
        )

        # Count topics
        topic_counts = [0] * n_total_topics
        for sample in dataset_train:
            for word in sample:
                topic = word.item() // n_words_per_topic
                topic_counts[topic] += 1
        results[dist] = topic_counts
        axes[i].bar(range(n_total_topics), topic_counts)
        axes[i].set_title(f"{dist} distribution")
        axes[i].set_xlabel("Topic")
        axes[i].set_ylabel("Count")

    plt.tight_layout()
    # plt.show()
    plt.savefig("topic_distributions.png")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("topic_distributions.csv", index=False)


# if __name__ == "__main__":
#     seed = random.randint(0, 1000000)
#     dataset_train, dataset_eval, dataset_test = get_lda_dataset(
#         seed=seed,
#         n_samples=20,
#         n_local_topics=1,
#         n_total_topics=10,
#         n_words_per_topic=7,
#         seq_len=20,
#     )
#     print(dataset_train[0])

# if __name__ == "__main__":
#     # Test normal distribution
#     import matplotlib.pyplot as plt

#     # Generate numbers from 0-100 using normal distribution
#     test_list = list(range(100))
#     n_samples = 10000
#     numbers = RandomGenerator.get_numbers_from_a_list_by_distribution(
#         n_numbers=n_samples, number_list=test_list, distribution="binomial"
#     )

#     # Plot histogram
#     plt.figure(figsize=(10, 6))
#     plt.hist(numbers, bins=50, density=True, alpha=0.7, color="skyblue")
#     plt.title("Normal Distribution Test")
#     plt.xlabel("Value")
#     plt.ylabel("Frequency")
#     plt.grid(True, alpha=0.3)
#     plt.show()
