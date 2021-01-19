# CGAN-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation
of [Conditional Generative Adversarial Networks](http://xxx.itp.ac.cn/pdf/1411.1784).

### Table of contents

1. [About Conditional Generative Adversarial Networks](#about-conditional-generative-adversarial-networks)
2. [Model Description](#model-description)
3. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download pretrained weights](#download-pretrained-weights-eg-mnist)
4. [Test](#test)
    * [Torch Hub call](#torch-hub-call)
    * [Base call](#base-call)
5. [Train](#train-eg-mnist)
6. [Contributing](#contributing)
7. [Credit](#credit)

### About Conditional Generative Adversarial Networks

If you're new to CGANs, here's an abstract straight from the paper:

Generative Adversarial Nets were recently introduced as a novel way to train generative models. In this work we
introduce the conditional version of generative adversarial nets, which can be constructed by simply feeding the data,
y, we wish to condition on to both the generator and discriminator. We show that this model can generate MNIST digits
conditioned on class labels. We also illustrate how this model could be used to learn a multi-modal model, and provide
preliminary examples of an application to image tagging in which we demonstrate how this approach can generate
descriptive tags which are not part of training labels.

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. It receives
a random noise z and generates images from this noise, which is called G(z).Discriminator is a discriminant network that
discriminates whether an image is real. The input is x, x is a picture, and the output is D of x is the probability that
x is a real picture, and if it's 1, it's 100% real, and if it's 0, it's not real.

### Installation

#### Clone and install requirements

```bash
$ git clone https://github.com/Lornatang/CGAN-PyTorch.git
$ cd CGAN-PyTorch/
$ pip3 install -r requirements.txt
```

#### Download pretrained weights (e.g. mnist)

```bash
$ cd weights/
$ python3 download_weights.py
```

### Test

#### Torch hub call

```python
# Using Torch Hub library.
import torch
import torchvision.utils as vutils

# Choose to use the device.
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the model into the specified device.
model = torch.hub.load("Lornatang/CGAN-PyTorch", "mnist", pretrained=True, verbose=False)
model.eval()
model = model.to(device)

# Create random noise image.
num_images = 64
number = 1
conditional = torch.randint(number, number + 1, (num_images,), device=device)
noise = torch.randn(num_images, 100, device=device)

# The noise is input into the generator model to generate the image.
with torch.no_grad():
    generated_images = model(noise, conditional)

# Save generate image.
vutils.save_image(generated_images, "mnist.png", normalize=True)
```

#### Base call

Using pre training model to generate pictures.

```text
usage: test.py [-h] [-a ARCH] [--num-classes NUM_CLASSES]
               [--number {0,1,2,3,4,5,6,7,8,9}] [-n NUM_IMAGES] [--outf PATH]
               [--device DEVICE]

An implementation of CGAN algorithm using PyTorch framework.

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: _gan | discriminator |
                        load_state_dict_from_url | mnist (default: mnist)
  --num-classes NUM_CLASSES
                        Number of classes for dataset. (default: 10).
  --number {0,1,2,3,4,5,6,7,8,9}
                        Specifies the generated number. (default: 1).
  -n NUM_IMAGES, --num-images NUM_IMAGES
                        How many samples are generated at one time. (default:
                        64).
  --outf PATH           The location of the image in the evaluation process.
                        (default: ``test``).
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default:
                        ``cpu``).

# Example (e.g. MNIST)
$ python3 test.py -a mnist --number 1
```

<span align="center"><img src="assets/mnist.gif" alt="">
</span>

### Train (e.g. MNIST)

```text
usage: train.py [-h] --dataset {mnist} [-a ARCH] [-j N] [--start-iter N]
                [--iters N] [-b N] [--lr LR] [--image-size IMAGE_SIZE]
                [--channels CHANNELS] [--num-classes NUM_CLASSES]
                [--pretrained] [--netD PATH] [--netG PATH]
                [--manualSeed MANUALSEED] [--device DEVICE]
                DIR

An implementation of GAN algorithm using PyTorch framework.

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --dataset {mnist}     mnist.
  -a ARCH, --arch ARCH  model architecture: _gan | discriminator |
                        load_state_dict_from_url | mnist (default: mnist)
  -j N, --workers N     Number of data loading workers. (default:8)
  --start-iter N        manual iter number (useful on restarts)
  --iters N             The number of iterations is needed in the training of
                        model. (default: 50000)
  -b N, --batch-size N  mini-batch size (default: 64), this is the total batch
                        size of all GPUs on the current node when using Data
                        Parallel or Distributed Data Parallel.
  --lr LR               Learning rate. (default:0.0002)
  --image-size IMAGE_SIZE
                        The height / width of the input image to network.
                        (default: 28).
  --channels CHANNELS   The number of channels of the image. (default: 1).
  --num-classes NUM_CLASSES
                        Number of classes for dataset. (default: 10).
  --pretrained          Use pre-trained model.
  --netD PATH           Path to latest discriminator checkpoint. (default:
                        ````).
  --netG PATH           Path to latest generator checkpoint. (default: ````).
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:1111)
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default:
                        ``0``).

# Example (e.g. MNIST)
$ python3 train.py data --dataset mnist -a mnist --image-size 28 --channels 1 --num-classes 10 --pretrained --device 0
```

If you want to load weights that you've trained before, run the following command.

```bash
$ python3 train.py data \
                   --dataset mnist \
                   -a mnist \
                   --image-size 28 \
                   --channels 1 \
                   --num-classes 10 \
                   --start-iter 10000 \
                   --netG weights/mnist_G_iter_10000.pth \
                   --netD weights/mnist_D_iter_10000.pth \
                   --device 0
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Conditional Generative Adversarial Networks

*Mehdi Mirza, Simon Osindero*

**Abstract**

Generative Adversarial Nets were recently introduced as a novel way to train generative models. In this work we
introduce the conditional version of generative adversarial nets, which can be constructed by simply feeding the data,
y, we wish to condition on to both the generator and discriminator. We show that this model can generate MNIST digits
conditioned on class labels. We also illustrate how this model could be used to learn a multi-modal model, and provide
preliminary examples of an application to image tagging in which we demonstrate how this approach can generate
descriptive tags which are not part of training labels.

[[Paper]](http://xxx.itp.ac.cn/pdf/1411.1784)

```
@article{DBLP:journals/corr/MirzaO14,
  author    = {Mehdi Mirza and
               Simon Osindero},
  title     = {Conditional Generative Adversarial Nets},
  journal   = {CoRR},
  volume    = {abs/1411.1784},
  year      = {2014},
  url       = {http://arxiv.org/abs/1411.1784},
  archivePrefix = {arXiv},
  eprint    = {1411.1784},
  timestamp = {Mon, 13 Aug 2018 16:48:15 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/MirzaO14.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```