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
    # In order to make the model repeatable, the first step is to set random seeds, and the second step is to set convolution algorithm.
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build a super-resolution model, if model_ If path is defined, the specified model weight will be loaded.
    model = configure(args)
    # If special choice model path.
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device("cpu")))
    # Switch model to eval mode.
    model.eval()

    # If the GPU is available, load the model into the GPU memory. This speed.
    if not torch.cuda.is_available():
        logger.warning("Using CPU, this will be slow.")
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # Setting this flag allows the built-in auto tuner of cudnn to automatically find the most efficient algorithm suitable
        # for the current configuration, so as to optimize the operation efficiency.
        cudnn.benchmark = True
        # Ensure that every time the same input returns the same result.
        cudnn.deterministic = True

    noise = torch.randn([args.num_images, 100])
    conditional = torch.randint(args.number, args.number + 1, (args.num_images,))
    # Move data to special device.
    if args.gpu is not None:
        noise = noise.cuda(args.gpu, non_blocking=True)
        conditional = conditional.cuda(args.gpu, non_blocking=True)

    # It only needs to reconstruct the low resolution image without the gradient information of the reconstructed image.
    with torch.no_grad():
        generated_images = model(noise, conditional)

    vutils.save_image(generated_images, os.path.join("tests", "test.png"), normalize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arch", metavar="ARCH", default="cgan",
                        choices=model_names,
                        help="model architecture: " +
                             " | ".join(model_names) +
                             ". (Default: `cgan`)")
    parser.add_argument("--number", type=int, default=1, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help="Specifies the generated number. (Default: 1)")
    parser.add_argument("--num-images", type=int, default=64,
                        help="How many samples are generated at one time. (Default: 64)")
    parser.add_argument("--model-path", default=None, type=str, metavar="PATH",
                        help="Path to latest checkpoint for model.")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                        help="Use pre-trained model.")
    parser.add_argument("--seed", default=666, type=int,
                        help="Seed for initializing training. (Default: 666)")
    parser.add_argument("--gpu", default=None, type=int,
                        help="GPU id to use.")
    args = parser.parse_args()

    print("##################################################\n")
    print("Run Testing Engine.\n")

    create_folder("tests")

    logger.info("TestEngine:")
    print("\tAPI version .......... 0.2.0")
    print("\tBuild ................ 2021.05.31")
    print("##################################################\n")
    main(args)

    logger.info("Test single image performance evaluation completed successfully.\n")
