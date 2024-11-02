import torchvision
import torchvision.transforms as transforms
import torch


def get_transforms(dataset: str):
    transform_train = None
    transform_test = None
    if dataset == "fashionmnist" or dataset == "mnist":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    if dataset == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    if dataset == "cifar100":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )

    assert transform_test is not None and transform_train is not None, (
        "Error, no dataset %s" % dataset
    )
    return transform_train, transform_test


def get_dataset(dataset, root="./data"):
    transform_train, transform_test = get_transforms(dataset)
    trainset, testset = None, None
    if dataset == "mnist":
        trainset = torchvision.datasets.MNIST(
            root=root, train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.MNIST(
            root=root, train=False, download=True, transform=transform_test
        )
    if dataset == "fashionmnist":
        trainset = torchvision.datasets.FashionMNIST(
            root=root, train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.FashionMNIST(
            root=root, train=False, download=True, transform=transform_test
        )
    if dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform_test
        )
    if dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(
            root=root, train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR100(
            root=root, train=False, download=True, transform=transform_test
        )

    return trainset, testset, testset


def get_minist_dataset(root="data"):
    trainset, testset, testset = get_dataset("mnist", root)
    return trainset, testset, testset


def get_fashionmnist_dataset(root="data"):
    trainset, testset, testset = get_dataset("fashionmnist", root)
    return trainset, testset, testset


def get_cifar10_dataset(root="data"):
    trainset, testset, testset = get_dataset("cifar10", root)
    return trainset, testset, testset


def get_cifar100_dataset(root="data"):
    trainset, testset, testset = get_dataset("cifar100", root)
    return trainset, testset, testset
