# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = [
    "Discriminator", "Generator", "discriminator",
    "mnist"
]

model_urls = {
    "mnist": "https://github.com/Lornatang/CGAN-PyTorch/releases/download/0.1.0/mnist-fa290ecd.pth",
}


class Discriminator(nn.Module):
    r""" An Discriminator model.

    `Generative Adversarial Networks model architecture from the One weird trick...
    <https://arxiv.org/abs/1411.1784>`_ paper.
    """

    def __init__(self, image_size: int = 28, channels: int = 1, num_classes: int = 10):
        """
        Args:
            image_size (int): The size of the image. (Default: 28).
            channels (int): The channels of the image. (Default: 1).
            num_classes (int): Number of classes for dataset. (default: 10).
        """
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            nn.Linear(channels * image_size * image_size + num_classes, 512),
            nn.LeakyReLU(0.2, True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input: torch.Tensor, labels: list = None) -> torch.Tensor:
        r""" Defines the computation performed at every call.

        Args:
            input (tensor): input tensor into the calculation.
            labels (list):  input tensor label.

        Returns:
            A four-dimensional vector (NCHW).
        """
        input = torch.flatten(input, 1)
        conditional = self.label_embedding(labels)
        conditional_input = torch.cat([input, conditional], dim=-1)
        out = self.main(conditional_input)
        return out


class Generator(nn.Module):
    r""" An Generator model.

    `Generative Adversarial Networks model architecture from the One weird trick...
    <https://arxiv.org/abs/1411.1784>`_ paper.
    """

    def __init__(self, image_size: int = 28, channels: int = 1, num_classes: int = 10):
        """
        Args:
            image_size (int): The size of the image. (Default: 28).
            channels (int): The channels of the image. (Default: 1).
            num_classes (int): Number of classes for dataset. (default: 10).
        """
        super(Generator, self).__init__()
        self.image_size = image_size
        self.channels = channels

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            nn.Linear(100 + num_classes, 128),
            nn.LeakyReLU(0.2, True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, True),

            nn.Linear(1024, channels * image_size * image_size),
            nn.Tanh()
        )

    def forward(self, input: torch.Tensor, labels: list = None) -> torch.Tensor:
        r""" Defines the computation performed at every call.

        Args:
            input (tensor): input tensor into the calculation.
            labels (list):  input tensor label.

        Returns:
            A four-dimensional vector (NCHW).
        """
        conditional_input = torch.cat([input, self.label_embedding(labels)], dim=-1)
        out = self.main(conditional_input)
        out = out.reshape(out.size(0), self.channels, self.image_size, self.image_size)
        return out


def _gan(arch, image_size, channels, num_classes, pretrained, progress):
    model = Generator(image_size, channels, num_classes)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress,
                                              map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    return model


def discriminator(**kwargs) -> Discriminator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1411.1784>`_ paper.
    """
    model = Discriminator(**kwargs)
    return model


def mnist(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1411.1784>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gan("mnist", 28, 1, 10, pretrained, progress)
