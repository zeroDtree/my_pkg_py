from .nature_language import (
    load_alpaca_gpt4,
    load_codefeedback,
    load_gsm8k,
    load_meta_math,
    load_sst2,
    load_wizardlm,
)
from .MT19937 import (
    load_mt19937,
    load_mt19937_8bits,
    load_mt19937_12bits,
    load_mt19937_32bits,
)
from .regular_language import get_regular_language_dataset
from .minist_cifar import (
    get_minist_dataset,
    get_fashionmnist_dataset,
    get_cifar10_dataset,
    get_cifar100_dataset,
)
from my_model.diagonal_network import get_diagonal_dataset
from .iris import get_iris_dataset


# return (train_set, validation_set, test_set)
DATASETMAPPING = {
    # language dataset ===========================================
    "alpaca_gpt4": load_alpaca_gpt4,
    "codefeedback": load_codefeedback,
    "gsm8k": load_gsm8k,
    "meta_math": load_meta_math,
    "sst2": load_sst2,
    "wizardlm": load_wizardlm,
    "mt19937": load_mt19937,
    "mt19937-8": load_mt19937_8bits,
    "mt19937-12": load_mt19937_12bits,
    "mt19937-32": load_mt19937_32bits,
    "regular_language": get_regular_language_dataset,
    # feature dataset ===========================================
    "diagonal": get_diagonal_dataset,
    "iris": get_iris_dataset,
    # image dataset =============================================
    "mnist": get_minist_dataset,
    "fashionmnist": get_fashionmnist_dataset,
    "cifar10": get_cifar10_dataset,
    "cifar100": get_cifar100_dataset,
}
