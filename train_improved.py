# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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
"""
Improved RCAN training script with CBAM attention and composite loss

This script demonstrates how to train RCAN with the following improvements:
1. CBAM (Convolutional Block Attention Module) for better spatial and channel attention
2. Composite loss function combining L1, perceptual, and adversarial losses

Usage:
    python train_improved.py

Note: Make sure to set the appropriate configuration in config.py:
    - mode = "train"
    - model_arch_name = "rcan_cbam_x4"  # for CBAM version
    - use_composite_loss = True  # enable composite loss
"""

import os
import time

import torch
from torch import nn, optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
import model
from dataset import CUDAPrefetcher, TrainImageDataset, TestImageDataset
from test import test
from utils import build_iqa_model, load_state_dict, make_directory, save_checkpoint, AverageMeter, ProgressMeter
from losses import CompositeLoss, ESRGANDiscriminator, relativistic_discriminator_loss


def main():
    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0
    best_ssim = 0.0

    train_data_prefetcher, test_data_prefetcher = load_dataset(config.train_gt_images_dir,
                                                               config.train_gt_image_size,
                                                               config.test_gt_images_dir,
                                                               config.test_lr_images_dir,
                                                               config.upscale_factor,
                                                               config.batch_size,
                                                               config.num_workers,
                                                               config.device)
    print("Load all datasets successfully.")

    sr_model, ema_sr_model = build_model(config.model_arch_name, config.device)
    print(f"Build `{config.model_arch_name}` model successfully.")

    # Build discriminator for adversarial training (if using composite loss)
    discriminator = None
    if hasattr(config, 'use_composite_loss') and config.use_composite_loss:
        discriminator = ESRGANDiscriminator().to(config.device)
        print("Build discriminator for adversarial training.")

    # Define loss function
    if hasattr(config, 'use_composite_loss') and config.use_composite_loss:
        criterion = CompositeLoss(
            l1_weight=getattr(config, 'l1_weight', 1.0),
            perceptual_weight=getattr(config, 'perceptual_weight', 0.006),
            adversarial_weight=getattr(config, 'adversarial_weight', 0.001)
        ).to(config.device)
        print("Using composite loss (L1 + Perceptual + Adversarial).")
    else:
        criterion = define_loss(config.device)
        print("Using L1 loss.")

    optimizer = define_optimizer(sr_model)
    print("Define all optimizer functions successfully.")

    # Define discriminator optimizer if using adversarial training
    discriminator_optimizer = None
    if discriminator is not None:
        discriminator_optimizer = optim.Adam(discriminator.parameters(), 
                                           lr=getattr(config, 'discriminator_lr', 1e-4),
                                           betas=(0.9, 0.999))
        print("Define discriminator optimizer successfully.")

    scheduler = define_scheduler(optimizer)
    print("Define all optimizer scheduler functions successfully.")

    # Create an IQA evaluation model
    psnr_model, ssim_model = build_iqa_model(config.upscale_factor, config.only_test_y_channel, config.device)

    print("Check whether to load pretrained model weights...")
    if config.pretrained_model_weights_path:
        sr_model = load_state_dict(sr_model, config.pretrained_model_weights_path)
        print(f"Loaded `{config.pretrained_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    print("Check whether the resume model is restored...")
    if config.resume_model_weights_path:
        sr_model, ema_sr_model, start_epoch, best_psnr, best_ssim, optimizer, scheduler = load_state_dict(
            sr_model,
            config.resume_model_weights_path,
            ema_sr_model,
            optimizer,
            scheduler,
            "resume")
        print("Loaded resume model weights.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # Create a experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    for epoch in range(start_epoch, config.epochs):
        if discriminator is not None:
            train_with_adversarial(sr_model,
                                 ema_sr_model,
                                 discriminator,
                                 train_data_prefetcher,
                                 criterion,
                                 optimizer,
                                 discriminator_optimizer,
                                 epoch,
                                 scaler,
                                 writer,
                                 config.device,
                                 config.train_print_frequency)
        else:
            train(sr_model,
                  ema_sr_model,
                  train_data_prefetcher,
                  criterion,
                  optimizer,
                  epoch,
                  scaler,
                  writer,
                  config.device,
                  config.train_print_frequency)
        
        psnr, ssim = test(sr_model,
                          test_data_prefetcher,
                          psnr_model,
                          ssim_model,
                          config.device,
                          config.test_print_frequency)

        # Write the evaluation results to the tensorboard
        writer.add_scalar(f"Test/PSNR", psnr, epoch + 1)
        writer.add_scalar(f"Test/SSIM", ssim, epoch + 1)

        print("\n")

        # Update LR
        scheduler.step()

        # Automatically save the model with the highest index
        is_best = psnr > best_psnr and ssim > best_ssim
        is_last = (epoch + 1) == config.epochs
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        
        checkpoint_dict = {
            "epoch": epoch + 1,
            "best_psnr": best_psnr,
            "best_ssim": best_ssim,
            "state_dict": sr_model.state_dict(),
            "ema_state_dict": ema_sr_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }
        
        # Add discriminator state if using adversarial training
        if discriminator is not None:
            checkpoint_dict["discriminator"] = discriminator.state_dict()
            checkpoint_dict["discriminator_optimizer"] = discriminator_optimizer.state_dict()
        
        save_checkpoint(checkpoint_dict,
                        f"epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "best.pth.tar",
                        "last.pth.tar",
                        is_best,
                        is_last)


def load_dataset(
        train_gt_images_dir: str,
        train_gt_image_size: int,
        test_gt_images_dir: str,
        test_lr_images_dir: str,
        upscale_factor: int,
        batch_size: int,
        num_workers: int,
        device: torch.device,
) -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test datasets
    train_datasets = TrainImageDataset(train_gt_images_dir, train_gt_image_size, upscale_factor)
    test_datasets = TestImageDataset(test_gt_images_dir, test_lr_images_dir)

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_data_prefetcher = CUDAPrefetcher(train_dataloader, device)
    test_data_prefetcher = CUDAPrefetcher(test_dataloader, device)

    return train_data_prefetcher, test_data_prefetcher


def build_model(model_arch_name: str, device: torch.device) -> [nn.Module, nn.Module]:
    sr_model = model.__dict__[model_arch_name]()
    sr_model = sr_model.to(device)
    ema_sr_model = AveragedModel(sr_model)
    ema_sr_model = ema_sr_model.to(device)

    return sr_model, ema_sr_model


def define_loss(device) -> nn.L1Loss:
    criterion = nn.L1Loss()
    criterion = criterion.to(device)

    return criterion


def define_optimizer(sr_model: nn.Module) -> optim.Adam:
    optimizer = optim.Adam(sr_model.parameters(),
                          config.model_lr,
                          config.model_betas,
                          config.model_eps,
                          config.model_weight_decay)

    return optimizer


def define_scheduler(optimizer) -> lr_scheduler.StepLR:
    scheduler = lr_scheduler.StepLR(optimizer,
                                   config.lr_scheduler_step_size,
                                   config.lr_scheduler_gamma)

    return scheduler


def train(
        sr_model: nn.Module,
        ema_sr_model: nn.Module,
        train_data_prefetcher: CUDAPrefetcher,
        criterion: nn.L1Loss,
        optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        device: torch.device = torch.device("cpu"),
        print_frequency: int = 1,
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_data_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    sr_model.train()
    ema_sr_model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_data_prefetcher.reset()
    batch_data = train_data_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        gt = batch_data["gt"].to(device, non_blocking=True)
        lr = batch_data["lr"].to(device, non_blocking=True)

        # Initialize generator gradients
        sr_model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            sr = sr_model(lr)
            loss = criterion(sr, gt)

        # Backpropagation
        scaler.scale(loss).backward()
        # update generator weights
        scaler.step(optimizer)
        scaler.update()

        # Update EMA
        ema_sr_model.update_parameters(sr_model)

        # Statistical loss value for terminal data output
        losses.update(loss.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % print_frequency == 0:
            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            progress.display(batch_index + 1)

        # Preload the next batch of data
        batch_data = train_data_prefetcher.next()

        # Add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1


def train_with_adversarial(
        sr_model: nn.Module,
        ema_sr_model: nn.Module,
        discriminator: nn.Module,
        train_data_prefetcher: CUDAPrefetcher,
        criterion: CompositeLoss,
        optimizer: optim.Adam,
        discriminator_optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        device: torch.device = torch.device("cpu"),
        print_frequency: int = 1,
) -> None:
    """Training function with adversarial loss"""
    # Calculate how many batches of data are in each Epoch
    batches = len(train_data_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    g_losses = AverageMeter("G_Loss", ":6.6f")
    d_losses = AverageMeter("D_Loss", ":6.6f")
    progress = ProgressMeter(batches, [batch_time, data_time, g_losses, d_losses], prefix=f"Epoch: [{epoch + 1}]")

    # Put the networks in training mode
    sr_model.train()
    ema_sr_model.train()
    discriminator.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_data_prefetcher.reset()
    batch_data = train_data_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        gt = batch_data["gt"].to(device, non_blocking=True)
        lr = batch_data["lr"].to(device, non_blocking=True)

        # Train Discriminator
        discriminator.zero_grad(set_to_none=True)
        
        with amp.autocast():
            sr = sr_model(lr).detach()  # Detach to avoid generator gradients
            real_preds = discriminator(gt)
            fake_preds = discriminator(sr)
            d_loss = relativistic_discriminator_loss(real_preds, fake_preds)
        
        scaler.scale(d_loss).backward()
        scaler.step(discriminator_optimizer)
        
        # Train Generator
        sr_model.zero_grad(set_to_none=True)
        
        with amp.autocast():
            sr = sr_model(lr)
            real_preds = discriminator(gt).detach()  # Detach to avoid discriminator gradients
            fake_preds = discriminator(sr)
            
            # Calculate composite loss
            loss_dict = criterion(sr, gt, real_preds, fake_preds)
            g_loss = loss_dict['total']

        # Backpropagation
        scaler.scale(g_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update EMA
        ema_sr_model.update_parameters(sr_model)

        # Statistical loss values for terminal data output
        g_losses.update(g_loss.item(), lr.size(0))
        d_losses.update(d_loss.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % print_frequency == 0:
            # Record losses during training and output to file
            writer.add_scalar("Train/G_Loss", g_loss.item(), batch_index + epoch * batches + 1)
            writer.add_scalar("Train/D_Loss", d_loss.item(), batch_index + epoch * batches + 1)
            writer.add_scalar("Train/L1_Loss", loss_dict['l1'].item(), batch_index + epoch * batches + 1)
            writer.add_scalar("Train/Perceptual_Loss", loss_dict['perceptual'].item(), batch_index + epoch * batches + 1)
            writer.add_scalar("Train/Adversarial_Loss", loss_dict['adversarial'].item(), batch_index + epoch * batches + 1)
            progress.display(batch_index + 1)

        # Preload the next batch of data
        batch_data = train_data_prefetcher.next()

        # Add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1


if __name__ == "__main__":
    main()