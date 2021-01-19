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

"""File for accessing GAN via PyTorch Hub https://pytorch.org/hub/
Usage:
    import torch
    import torchvision.utils as vutils

    # Choose to use the device.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the model into the specified device.
    model = torch.hub.load("Lornatang/CGAN-PyTorch", "mnist", pretrained=True, verbose=False)
    model.eval()
    model = model.to(device)
"""
import torch
from torch.hub import load_state_dict_from_url

from cgan_pytorch.models import Generator

model_urls = {
    "mnist": "https://github.com/Lornatang/CGAN-PyTorch/releases/download/0.1.0/CGAN_mnist-8ee70a39.pth",
}

dependencies = ["torch"]


def create(arch, image_size, channels, num_classes, pretrained, progress):
    """ Creates a specified GAN model

    Args:
        arch (str): Arch name of model.
        image_size (int): Number of image size.
        channels (int): Number of input channels.
        num_classes (int): Number of classes for dataset.
        pretrained (bool): Load pretrained weights into the model.
        progress (bool): Show progress bar when downloading weights.

    Returns:
        PyTorch model.
    """
    model = Generator(image_size, channels, num_classes)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress,
                                              map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    return model


def mnist(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1406.2661>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return create("mnist", 28, 1, 10, pretrained, progress)
