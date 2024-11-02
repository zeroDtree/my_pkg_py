from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torch


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def FashionMnistNeuralNetwork():
    return NeuralNetwork()


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_relu_stack = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1
            ),  # 输入通道1，输出通道32，卷积核大小3x3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2最大池化
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1
            ),  # 输入通道32，输出通道64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2最大池化
        )
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),  # 根据卷积和池化后的输出尺寸进行调整
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.conv_relu_stack(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def FashionMnistConvolutionalNeuralNetwork():
    return ConvolutionalNeuralNetwork()


class Cifar10Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 3,32,32 -> 8,32,32
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1
        )
        # 8,32,32 -> 1,16,16
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=1
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1 * 16 * 16, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def Cifar10NeuralNetwork():
    return Cifar10Net()


class MnistNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        channels = 1
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=channels, kernel_size=2, stride=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=2, stride=2
        )
        self.fc1 = nn.Linear(channels * 7 * 7, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class MiniResNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, out_channels=1):
        super(MiniResNet, self).__init__()
        self.in_channels = in_channels
        # 1,28,28 -> 1,28,28
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 1,28,28 -> 1,28,28
        self.layer1 = self._make_layer(out_channels, 1)
        # 1,28,28 -> 1,28,28
        self.layer2 = self._make_layer(out_channels, 1)
        # 1,28，28 -> 1,7,7
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(7 * 7 * out_channels, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class CifarMiniResNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, out_channels=1):
        super(CifarMiniResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(out_channels, 1)
        self.layer2 = self._make_layer(out_channels, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(8 * 8 * out_channels, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(out_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
