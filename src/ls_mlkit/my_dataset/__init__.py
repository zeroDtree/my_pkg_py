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
    load_mt19937_16bits,
    load_mt19937_32bits,
    load_mt19937_8bits_with_eval,
    load_mt19937_12bits_with_eval,
    load_mt19937_16bits_with_eval,
    load_mt19937_32bits_with_eval,
)
from .regular_language import get_regular_language_dataset
from .minist_cifar import (
    get_minist_dataset,
    get_fashionmnist_dataset,
    get_cifar10_dataset,
    get_cifar100_dataset,
)
from .iris import get_iris_dataset
from .lda_dataset import get_lda_dataset


class DATASET_NAME:
    # language dataset ===========================================
    ALPACA_GPT4 = "alpaca_gpt4"
    CODEFEEDBACK = "codefeedback"
    GSM8K = "gsm8k"
    META_MATH = "meta_math"
    SST2 = "sst2"
    WIZARDLM = "wizardlm"
    MT19937 = "mt19937"
    MT19937_8 = "mt19937-8"
    MT19937_12 = "mt19937-12"
    MT19937_16 = "mt19937-16"
    MT19937_32 = "mt19937-32"
    MT19937_8_EVAL = "mt19937-8-eval"
    MT19937_12_EVAL = "mt19937-12-eval"
    MT19937_16_EVAL = "mt19937-16-eval"
    MT19937_32_EVAL = "mt19937-32-eval"
    REGULAR_LANGUAGE = "regular_language"
    LDA = "lda"
    # feature dataset ===========================================
    IRIS = "iris"
    # image dataset =============================================
    MNIST = "mnist"
    FASHIONMNIST = "fashionmnist"
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"


# return (train_set, validation_set, test_set)
DATASET_MAPPING = {
    # language dataset ===========================================
    DATASET_NAME.ALPACA_GPT4: load_alpaca_gpt4,
    DATASET_NAME.CODEFEEDBACK: load_codefeedback,
    DATASET_NAME.GSM8K: load_gsm8k,
    DATASET_NAME.META_MATH: load_meta_math,
    DATASET_NAME.SST2: load_sst2,
    DATASET_NAME.WIZARDLM: load_wizardlm,
    DATASET_NAME.MT19937: load_mt19937,
    DATASET_NAME.MT19937_8: load_mt19937_8bits,
    DATASET_NAME.MT19937_12: load_mt19937_12bits,
    DATASET_NAME.MT19937_16: load_mt19937_16bits,
    DATASET_NAME.MT19937_32: load_mt19937_32bits,
    DATASET_NAME.MT19937_8_EVAL: load_mt19937_8bits_with_eval,
    DATASET_NAME.MT19937_12_EVAL: load_mt19937_12bits_with_eval,
    DATASET_NAME.MT19937_16_EVAL: load_mt19937_16bits_with_eval,
    DATASET_NAME.MT19937_32_EVAL: load_mt19937_32bits_with_eval,
    DATASET_NAME.REGULAR_LANGUAGE: get_regular_language_dataset,
    DATASET_NAME.LDA: get_lda_dataset,
    # feature dataset ===========================================
    DATASET_NAME.IRIS: get_iris_dataset,
    # image dataset =============================================
    DATASET_NAME.MNIST: get_minist_dataset,
    DATASET_NAME.FASHIONMNIST: get_fashionmnist_dataset,
    DATASET_NAME.CIFAR10: get_cifar10_dataset,
    DATASET_NAME.CIFAR100: get_cifar100_dataset,
}
