# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
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
import argparse
import logging
import os
import random
import warnings

import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

import cgan_pytorch.models as models
from cgan_pytorch.utils import configure
from cgan_pytorch.utils import create_folder

# Find all available models.
model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

# It is a convenient method for simple scripts to configure the log package at one time.
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def main(args):
    if args.seed is not None:
        # In order to make the model repeatable, the first step is to set random seeds, and the second step is to set convolution algorithm.
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        warnings.warn("You have chosen to seed testing. "
                      "This will turn on the CUDNN deterministic setting, "
                      "which can slow down your training considerably! "
                      "You may see unexpected behavior when restarting "
                      "from checkpoints.")
        # for the current configuration, so as to optimize the operation efficiency.
        cudnn.benchmark = True
        # Ensure that every time the same input returns the same result.
        cudnn.deterministic = True

    # Build a super-resolution model, if model_ If path is defined, the specified model weight will be loaded.
    model = configure(args)
    # If special choice model path.
    if args.model_path is not None:
        logger.info(f"You loaded the specified weight. Load weights from `{os.path.abspath(args.model_path)}`.")
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device("cpu")))
    # Switch model to eval mode.
    model.eval()

    # If the GPU is available, load the model into the GPU memory. This speed.
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # Randomly generate a Gaussian noise map.
    logger.info("Randomly generate a Gaussian noise image.")
    noise = torch.randn([args.num_images, 100])
    conditional = torch.randint(args.conditional, args.conditional + 1, (args.num_images,))
    # Move data to special device.
    if args.gpu is not None:
        noise = noise.cuda(args.gpu)
        conditional = conditional.cuda(args.gpu)

    # It only needs to reconstruct the low resolution image without the gradient information of the reconstructed image.
    with torch.no_grad():
        logger.info("Generating...")
        generated_images = model(noise, conditional)

    save_path = os.path.join("tests", "test.png")
    logger.info(f"Saving image to `{save_path}`...")
    vutils.save_image(generated_images, save_path, normalize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="cgan", type=str, choices=model_names,
                        help="model architecture: " +
                             " | ".join(model_names) +
                             ". (Default: `cgan`)")
    parser.add_argument("--conditional", default=1, type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help="Specifies the generated conditional. (Default: 1)")
    parser.add_argument("--num-images", default=64, type=int,
                        help="How many samples are generated at one time. (Default: 64)")
    parser.add_argument("--model-path", default="weights/GAN-last.pth", type=str,
                        help="Path to latest checkpoint for model. (Default: `weights/GAN-last.pth`)")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                        help="Use pre-trained model.")
    parser.add_argument("--seed", default=None, type=int,
                        help="Seed for initializing testing.")
    parser.add_argument("--gpu", default=None, type=int,
                        help="GPU id to use.")
    args = parser.parse_args()

    print("##################################################\n")
    print("Run Testing Engine.\n")

    create_folder("tests")

    logger.info("TestEngine:")
    print("\tAPI version .......... 0.2.0")
    print("\tBuild ................ 2021.06.02")
    print("##################################################\n")
    main(args)

    logger.info("Test single image performance evaluation completed successfully.\n")
