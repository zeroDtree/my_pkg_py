from .image_classification import (
    NeuralNetwork,
    ConvolutionalNeuralNetwork,
    Cifar10Net,
    Cifar10NeuralNetwork,
    FashionMnistNeuralNetwork,
    FashionMnistConvolutionalNeuralNetwork,
    MnistNeuralNetwork,
)
from .diagonal_network import DiagonalNeuralNetwork, DiagonalNeuralNetworkDataset
from .ffn import FFNNeuralNetwork, IrisFFNNeuralNetwork
from .dnnODEModel import DNNClassificationODEModel
from .longLinear import LongLinearModel
from .decoder_tf import (
    greedy_decode,
    Tokenizer,
    get_collate_fn,
    CausalLanguageModelConfig,
    CausalLanguageModel,
    CausalLanguageModelConfigForAuto,
    CausalLanguageModelForAuto,
    register_model as register_causal_model,
)
from .gpt2 import GPT2, GPT2Tokenizer
from .llama import get_causal_llama_model

from .model_factory import get_text_to_text_model
from .cifar import (
    alexnet as cifar_alexnet,
    densenet as cifar_densenet,
    resnet as cifar_resnet,
    vgg16_bn as cifar_vgg16_bn,
    vgg19_bn as cifar_vgg19_bn,
    wrn as cifar_wrn,
    preresnet as cifar_preresnet,
    mlp as cifar_mlp,
    pyramidnet as cifar_pyramidnet,
    get_network_cifar,
)
from .mnist import wrn as mnist_wrn, mlp as mnist_mlp, get_network_mnist


def get_model_mnist_cifar(cfg):
    from omegaconf import OmegaConf
    from my_model import get_network_mnist, get_network_cifar
    from omegaconf import DictConfig

    cfg: DictConfig

    nc = {
        "mnist": 10,
        "fashionmnist": 10,
        "cifar10": 10,
        "cifar100": 100,
    }

    net = None
    cfg.model.outputs_dim = nc[cfg.dataset.name.lower()]
    if cfg.dataset.name.lower() == "mnist" or cfg.dataset.name.lower() == "fashionmnist":
        print("dataset:", cfg.dataset.name.lower())
        if cfg.model.name == "wrn":
            net = get_network_mnist(
                cfg.model.name,
                depth=cfg.model.depth,
                num_classes=cfg.model.outputs_dim,
                growthRate=cfg.model.growthRate,
                compressionRate=cfg.model.compressionRate,
                widen_factor=cfg.model.widen_factor,
                dropRate=cfg.model.dropRate,
            )
        elif cfg.model.name == "mlp":
            net = get_network_mnist(cfg.model.name)
    if cfg.dataset.name.lower() == "cifar10" or cfg.dataset.name.lower() == "cifar100":
        print("dataset:", cfg.dataset.lower())
        if cfg.model.name == "preresnet":
            net = get_network_cifar(
                cfg.model.name, depth=cfg.model.depth, num_classes=cfg.model.outputs_dim
            )
        elif cfg.model.name == "pyramidnet":
            net = get_network_cifar(
                cfg.model.name,
                depth=cfg.model.depth,
                alpha=48,
                input_shape=(1, 3, 32, 32),
                num_classes=cfg.model.outputs_dim,
                base_channels=16,
                block_type="bottleneck",
            )
        else:
            net = get_network_cifar(
                cfg.model.name,
                depth=cfg.model.depth,
                num_classes=cfg.model.outputs_dim,
                growthRate=cfg.model.growthRate,
                compressionRate=cfg.model.compressionRate,
                widen_factor=cfg.model.widen_factor,
                dropRate=cfg.model.dropRate,
            )
    assert net is not None, "model is not found"
    return net


MODELMAPPING = {
    "Cifar10Net": Cifar10Net,
    "Cifar10NeuralNetwork": Cifar10NeuralNetwork,
    "FashionMnistNeuralNetwork": FashionMnistNeuralNetwork,
    "FashionMnistConvolutionalNeuralNetwork": FashionMnistConvolutionalNeuralNetwork,
    "DiagonalNeuralNetwork": DiagonalNeuralNetwork,
    "IrisFFNNeuralNetwork": IrisFFNNeuralNetwork,
    "MnistNeuralNetwork": MnistNeuralNetwork,
    # cifar ================================================================
    "cifar_alexnet": cifar_alexnet,
    "cifar_densenet": cifar_densenet,
    "cifar_resnet": cifar_resnet,
    "cifar_vgg16_bn": cifar_vgg16_bn,
    "cifar_vgg19_bn": cifar_vgg19_bn,
    "cifar_wrn": cifar_wrn,
    "cifar_preresnet": cifar_preresnet,
    "cifar_mlp": cifar_mlp,
    "cifar_pyramidnet": cifar_pyramidnet,
    # mnist ================================================================
    "mnist_wrn": mnist_wrn,
    "mnist_mlp": mnist_mlp,
}
