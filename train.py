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
import time
import warnings

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

import cgan_pytorch.models as models
from cgan_pytorch.models.discriminator import discriminator_for_mnist
from cgan_pytorch.utils.common import AverageMeter
from cgan_pytorch.utils.common import ProgressMeter
from cgan_pytorch.utils.common import configure
from cgan_pytorch.utils.common import create_folder

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
        warnings.warn("You have chosen to seed training. "
                      "This will turn on the CUDNN deterministic setting, "
                      "which can slow down your training considerably! "
                      "You may see unexpected behavior when restarting "
                      "from checkpoints.")
        # for the current configuration, so as to optimize the operation efficiency.
        cudnn.benchmark = True
        # Ensure that every time the same input returns the same result.
        cudnn.deterministic = True

    if args.gpu is not None:
        logger.warning("You have chosen a specific GPU. This will completely disable data parallelism.")

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, args=(ngpus_per_node, args), nprocs=ngpus_per_node)
    else:
        # Simply call main_worker function
        main_worker(ngpus_per_node, args)


def main_worker(ngpus_per_node, args):
    if args.gpu is not None:
        logger.info(f"Use GPU: {args.gpu} for training.")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    # create model
    generator = configure(args)
    discriminator = discriminator_for_mnist(args.image_size, args.channels)

    if not torch.cuda.is_available():
        logger.warning("Using CPU, this will be slow.")
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            discriminator.cuda(args.gpu)
            generator.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            discriminator = nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.gpu])
            generator = nn.parallel.DistributedDataParallel(generator, device_ids=[args.gpu])
        else:
            discriminator.cuda()
            generator.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            discriminator = nn.parallel.DistributedDataParallel(discriminator)
            generator = nn.parallel.DistributedDataParallel(generator)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        discriminator = discriminator.cuda(args.gpu)
        generator = generator.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            discriminator.features = torch.nn.DataParallel(discriminator.features)
            generator.features = torch.nn.DataParallel(generator.features)
            discriminator.cuda()
            generator.cuda()
        else:
            discriminator = torch.nn.DataParallel(discriminator).cuda()
            generator = torch.nn.DataParallel(generator).cuda()

    # Loss of original GAN paper.
    adversarial_criterion = nn.MSELoss().cuda(args.gpu)

    fixed_noise = torch.randn([args.batch_size, 100])
    fixed_conditional = torch.randint(0, 1, (args.batch_size,))
    if args.gpu is not None:
        fixed_noise = fixed_noise.cuda(args.gpu)
        fixed_conditional = fixed_conditional.cuda(args.gpu)

    # All optimizer function and scheduler function.
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Selection of appropriate treatment equipment.
    dataset = torchvision.datasets.MNIST(root=args.data, download=True,
                                         transform=transforms.Compose([
                                             transforms.Resize((args.image_size, args.image_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,))
                                         ]))

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=(sampler is None),
                                             pin_memory=True,
                                             sampler=sampler,
                                             num_workers=args.workers)

    # Load pre training model.
    if args.netD != "":
        discriminator.load_state_dict(torch.load(args.netD))
    if args.netG != "":
        generator.load_state_dict(torch.load(args.netG))

    # Create a SummaryWriter at the beginning of training.
    writer = SummaryWriter(f"runs/{args.arch}_logs")

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)

        batch_time = AverageMeter("Time", ":6.4f")
        d_losses = AverageMeter("D Loss", ":6.6f")
        g_losses = AverageMeter("G Loss", ":6.6f")
        d_x_losses = AverageMeter("D(x)", ":6.6f")
        d_g_z1_losses = AverageMeter("D(G(z1))", ":6.6f")
        d_g_z2_losses = AverageMeter("D(G(z2))", ":6.6f")

        progress = ProgressMeter(num_batches=len(dataloader),
                                 meters=[batch_time, d_losses, g_losses, d_x_losses, d_g_z1_losses, d_g_z2_losses],
                                 prefix=f"Epoch: [{epoch}]")

        # Switch to train mode.
        discriminator.train()
        generator.train()

        end = time.time()
        for i, (inputs, target) in enumerate(dataloader):
            # Move data to special device.
            if args.gpu is not None:
                inputs = inputs.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            batch_size = inputs.size(0)

            # The real sample label is 1, and the generated sample label is 0.
            real_label = torch.full((batch_size, 1), 1, dtype=inputs.dtype).cuda(args.gpu, non_blocking=True)
            fake_label = torch.full((batch_size, 1), 0, dtype=inputs.dtype).cuda(args.gpu, non_blocking=True)

            noise = torch.randn([batch_size, 100])
            conditional = torch.randint(0, 10, (batch_size,))
            # Move data to special device.
            if args.gpu is not None:
                noise = noise.cuda(args.gpu, non_blocking=True)
                conditional = conditional.cuda(args.gpu, non_blocking=True)

            ##############################################
            # (1) Update D network: max E(x)[log(D(x))] + E(z)[log(1- D(z))]
            ##############################################
            # Set discriminator gradients to zero.
            discriminator.zero_grad()

            # Train with real.
            real_output = discriminator(inputs, target)
            d_loss_real = adversarial_criterion(real_output, real_label)
            d_loss_real.backward()
            d_x = real_output.mean()

            # Train with fake.
            fake = generator(noise, conditional)
            fake_output = discriminator(fake.detach(), conditional)
            d_loss_fake = adversarial_criterion(fake_output, fake_label)
            d_loss_fake.backward()
            d_g_z1 = fake_output.mean()

            # Count all discriminator losses.
            d_loss = d_loss_real + d_loss_fake
            discriminator_optimizer.step()

            ##############################################
            # (2) Update G network: min E(z)[log(1- D(z))]
            ##############################################
            # Set generator gradients to zero.
            generator.zero_grad()

            fake_output = discriminator(fake, conditional)
            g_loss = adversarial_criterion(fake_output, real_label)
            g_loss.backward()
            d_g_z2 = fake_output.mean()
            generator_optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # measure accuracy and record loss
            d_losses.update(d_loss.item(), inputs.size(0))
            g_losses.update(g_loss.item(), inputs.size(0))
            d_x_losses.update(d_x.item(), inputs.size(0))
            d_g_z1_losses.update(d_g_z1.item(), inputs.size(0))
            d_g_z2_losses.update(d_g_z2.item(), inputs.size(0))

            iters = i + epoch * len(dataloader) + 1
            writer.add_scalar("Train/D_Loss", d_loss.item(), iters)
            writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
            writer.add_scalar("Train/D_x", d_x.item(), iters)
            writer.add_scalar("Train/D_G_z1", d_g_z1.item(), iters)
            writer.add_scalar("Train/D_G_z2", d_g_z2.item(), iters)

            # Output results every 100 batches.
            if i % 100 == 0:
                progress.display(i)

        # Each Epoch validates the model once.
        with torch.no_grad():
            # Switch model to eval mode.
            generator.eval()
            sr = generator(fixed_noise, fixed_conditional)
            vutils.save_image(sr.detach(), os.path.join("runs", f"GAN_epoch_{epoch}.png"), normalize=True)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            torch.save(generator.state_dict(), os.path.join("weights", f"Generator_epoch{epoch}.pth"))
            torch.save(discriminator.state_dict(), os.path.join("weights", f"Discriminator_epoch{epoch}.pth"))

    torch.save(generator.state_dict(), os.path.join("weights", f"GAN-last.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="cgan", type=str, choices=model_names,
                        help="Model architecture: " +
                             " | ".join(model_names) +
                             ". (Default: `cgan`)")
    parser.add_argument("data", metavar="DIR",
                        help="Path to dataset.")
    parser.add_argument("--workers", default=4, type=int,
                        help="Number of data loading workers. (Default: 4)")
    parser.add_argument("--epochs", default=128, type=int,
                        help="Number of total epochs to run. (Default: 128)")
    parser.add_argument("--start-epoch", default=0, type=int,
                        help="Manual epoch number (useful on restarts). (Default: 0)")
    parser.add_argument("-b", "--batch-size", default=64, type=int,
                        help="The batch size of the dataset. (Default: 64)")
    parser.add_argument("--lr", default=0.0002, type=float,
                        help="Learning rate. (Default: 0.0002)")
    parser.add_argument("--image-size", default=28, type=int,
                        help="Image size of high resolution image. (Default: 28)")
    parser.add_argument("--channels", default=1, type=int,
                        help="The number of channels of the image. (Default: 1)")
    parser.add_argument("--netD", default="", type=str,
                        help="Path to Discriminator checkpoint.")
    parser.add_argument("--netG", default="", type=str,
                        help="Path to Generator checkpoint.")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                        help="Use pre-trained model.")
    parser.add_argument("--world-size", default=-1, type=int,
                        help="Number of nodes for distributed training.")
    parser.add_argument("--rank", default=-1, type=int,
                        help="Node rank for distributed training. (Default: -1)")
    parser.add_argument("--dist-url", default="tcp://59.110.31.55:12345", type=str,
                        help="url used to set up distributed training. (Default: `tcp://59.110.31.55:12345`)")
    parser.add_argument("--dist-backend", default="nccl", type=str,
                        help="Distributed backend. (Default: `nccl`)")
    parser.add_argument("--seed", default=None, type=int,
                        help="Seed for initializing training.")
    parser.add_argument("--gpu", default=None, type=int,
                        help="GPU id to use.")
    parser.add_argument("--multiprocessing-distributed", action="store_true",
                        help="Use multi-processing distributed training to launch "
                             "N processes per node, which has N GPUs. This is the "
                             "fastest way to use PyTorch for either single node or "
                             "multi node data parallel training.")
    args = parser.parse_args()

    print("##################################################\n")
    print("Run Training Engine.\n")

    create_folder("runs")
    create_folder("weights")

    logger.info("TrainingEngine:")
    print("\tAPI version .......... 0.2.0")
    print("\tBuild ................ 2021.06.02")
    print("##################################################\n")

    main(args)

    logger.info("All training has been completed successfully.\n")
