import os
import time

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader

import config
import model
from dataset import TestImageDataset, CUDAPrefetcher
from imgproc import tensor_to_image
from utils import build_iqa_model, load_state_dict, make_directory, AverageMeter, ProgressMeter


def load_dataset(test_gt_images_dir: str, test_lr_images_dir: str, device: torch.device) -> CUDAPrefetcher:
    test_datasets = TestImageDataset(test_gt_images_dir, test_lr_images_dir)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=False)
    test_data_prefetcher = CUDAPrefetcher(test_dataloader, device)

    return test_data_prefetcher


def build_model(model_arch_name: str, device: torch.device) -> nn.Module:
    # Build model
    sr_model = model.__dict__[model_arch_name]()
    sr_model = sr_model.to(device=device)
    # Set the model to evaluation mode
    sr_model.eval()

    return sr_model


def test_with_save(
        sr_model: nn.Module,
        test_data_prefetcher: CUDAPrefetcher,
        psnr_model: nn.Module,
        ssim_model: nn.Module,
        save_dir: str,
        device: torch.device = torch.device("cpu"),
        print_frequency: int = 1,
) -> [float, float]:
    # The information printed by the progress bar
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(len(test_data_prefetcher), [batch_time, psnres, ssimes], prefix=f"Test: ")

    # Set the model as validation model
    sr_model.eval()

    # Initialize data batches
    batch_index = 0

    # Set the data set iterator pointer to 0 and load the first batch of data
    test_data_prefetcher.reset()
    batch_data = test_data_prefetcher.next()

    # Record the start time of verifying a batch
    end = time.time()

    while batch_data is not None:
        # Load batches of data
        gt = batch_data["gt"].to(device=device, non_blocking=True)
        lr = batch_data["lr"].to(device=device, non_blocking=True)

        # inference
        with torch.no_grad():
            sr = sr_model(lr)

        # Calculate the image IQA
        psnr = psnr_model(sr, gt)
        ssim = ssim_model(sr, gt)
        psnres.update(psnr.item(), lr.size(0))
        ssimes.update(ssim.item(), lr.size(0))

        # Save super-resolution image
        sr_image = tensor_to_image(sr, range_norm=False, half=False)
        sr_image_bgr = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        
        # Generate output filename
        output_filename = f"sr_image_{batch_index:04d}.png"
        output_path = os.path.join(save_dir, output_filename)
        
        # Save the image
        cv2.imwrite(output_path, sr_image_bgr)
        
        print(f"Saved: {output_path} | PSNR: {psnr.item():.2f} dB | SSIM: {ssim.item():.4f}")

        # Record the total time to verify a batch
        batch_time.update(time.time() - end)
        end = time.time()

        # Output a verification log information
        if batch_index % print_frequency == 0:
            progress.display(batch_index + 1)

        # Preload the next batch of data
        batch_data = test_data_prefetcher.next()

        # Add 1 to the number of data batches
        batch_index += 1

    # Print the performance index of the model at the current epoch
    progress.display_summary()

    return psnres.avg, ssimes.avg


def main() -> None:
    test_data_prefetcher = load_dataset(config.test_gt_images_dir, config.test_lr_images_dir, config.device)
    sr_model = build_model(config.model_arch_name, config.device)
    psnr_model, ssim_model = build_iqa_model(config.upscale_factor, config.only_test_y_channel, config.device)

    # Load the super-resolution model weights
    sr_model = load_state_dict(sr_model, config.model_weights_path)

    # Create a folder of super-resolution experiment results
    make_directory(config.test_sr_images_dir)

    psnr, ssim = test_with_save(sr_model,
                               test_data_prefetcher,
                               psnr_model,
                               ssim_model,
                               config.test_sr_images_dir,
                               config.device)

    print(f"\n=== Final Results ===")
    print(f"Average PSNR: {psnr:.2f} dB")
    print(f"Average SSIM: {ssim:.4f}")
    print(f"Super-resolution images saved to: {config.test_sr_images_dir}")


if __name__ == "__main__":
    main()