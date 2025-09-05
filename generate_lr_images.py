#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Low-Resolution Images from High-Resolution Images

This script converts high-resolution images to low-resolution images using bicubic downsampling.
It's designed to create LR images for super-resolution model testing.
"""

import os
import argparse
from typing import List
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image


def ensure_directory(directory: str) -> None:
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def get_image_files(directory: str) -> List[str]:
    """Get all image files from directory"""
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    image_files = []
    
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(supported_formats):
            image_files.append(filename)
    
    image_files.sort()  # Sort for consistent processing order
    return image_files


def downsample_image(image_path: str, scale_factor: int = 4, method: str = 'bicubic') -> np.ndarray:
    """Downsample image using specified method"""
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Get original dimensions
    height, width = image.shape[:2]
    
    # Calculate new dimensions
    new_height = height // scale_factor
    new_width = width // scale_factor
    
    # Choose interpolation method
    if method == 'bicubic':
        interpolation = cv2.INTER_CUBIC
    elif method == 'bilinear':
        interpolation = cv2.INTER_LINEAR
    elif method == 'nearest':
        interpolation = cv2.INTER_NEAREST
    else:
        raise ValueError(f"Unsupported interpolation method: {method}")
    
    # Downsample
    lr_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    
    return lr_image


def process_images(hr_dir: str, lr_dir: str, scale_factor: int = 4, 
                  method: str = 'bicubic', quality: int = 95) -> None:
    """Process all images in the HR directory"""
    
    # Ensure output directory exists
    ensure_directory(lr_dir)
    
    # Get all image files
    image_files = get_image_files(hr_dir)
    
    if not image_files:
        print(f"No image files found in {hr_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Scale factor: {scale_factor}x")
    print(f"Interpolation method: {method}")
    print(f"Input directory: {hr_dir}")
    print(f"Output directory: {lr_dir}")
    print("-" * 50)
    
    # Process each image
    successful = 0
    failed = 0
    
    for filename in tqdm(image_files, desc="Processing images"):
        try:
            # Full paths
            hr_path = os.path.join(hr_dir, filename)
            lr_path = os.path.join(lr_dir, filename)
            
            # Downsample image
            lr_image = downsample_image(hr_path, scale_factor, method)
            
            # Save LR image
            if filename.lower().endswith('.png'):
                # For PNG, use lossless compression
                cv2.imwrite(lr_path, lr_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            elif filename.lower().endswith(('.jpg', '.jpeg')):
                # For JPEG, use specified quality
                cv2.imwrite(lr_path, lr_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                # For other formats, use default settings
                cv2.imwrite(lr_path, lr_image)
            
            successful += 1
            
        except Exception as e:
            print(f"\nError processing {filename}: {str(e)}")
            failed += 1
            continue
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {successful} images")
    print(f"Failed: {failed} images")
    
    if successful > 0:
        # Show sample information
        sample_hr = os.path.join(hr_dir, image_files[0])
        sample_lr = os.path.join(lr_dir, image_files[0])
        
        if os.path.exists(sample_hr) and os.path.exists(sample_lr):
            hr_img = cv2.imread(sample_hr)
            lr_img = cv2.imread(sample_lr)
            
            print(f"\nSample image info ({image_files[0]}):")
            print(f"  HR size: {hr_img.shape[1]}x{hr_img.shape[0]}")
            print(f"  LR size: {lr_img.shape[1]}x{lr_img.shape[0]}")
            print(f"  Actual scale: {hr_img.shape[1]/lr_img.shape[1]:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Generate low-resolution images from high-resolution images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default 4x downsampling
  python generate_lr_images.py --hr_dir /path/to/hr/images --lr_dir /path/to/lr/images
  
  # Custom scale factor and method
  python generate_lr_images.py --hr_dir /path/to/hr --lr_dir /path/to/lr --scale 2 --method bilinear
  
  # For DIV2K dataset
  python generate_lr_images.py \
    --hr_dir "/data/rcan/RCAN-PyTorch-master/data/DIV2K/RCAN/valid" \
    --lr_dir "/data/rcan/RCAN-PyTorch-master/data/DIV2K/RCAN/LR" \
    --scale 4 --method bicubic
        """)
    
    parser.add_argument('--hr_dir', type=str, required=True,
                       help='Path to high-resolution images directory')
    parser.add_argument('--lr_dir', type=str, required=True,
                       help='Path to output low-resolution images directory')
    parser.add_argument('--scale', type=int, default=4, choices=[2, 3, 4, 8],
                       help='Downsampling scale factor (default: 4)')
    parser.add_argument('--method', type=str, default='bicubic', 
                       choices=['bicubic', 'bilinear', 'nearest'],
                       help='Interpolation method (default: bicubic)')
    parser.add_argument('--quality', type=int, default=95, 
                       help='JPEG quality for .jpg files (default: 95)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.hr_dir):
        print(f"Error: HR directory does not exist: {args.hr_dir}")
        return
    
    if args.quality < 1 or args.quality > 100:
        print(f"Error: Quality must be between 1 and 100, got {args.quality}")
        return
    
    # Process images
    try:
        process_images(args.hr_dir, args.lr_dir, args.scale, args.method, args.quality)
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()