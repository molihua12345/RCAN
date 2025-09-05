#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file for model comparison script
"""

import torch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configurations
MODEL_CONFIGS = {
    'rcan_x4': {
        'arch': 'rcan_x4',
        'weights_path': '/data/rcan/RCAN-PyTorch-master/results/pretrained_models/RCAN_x4-DIV2K-2dfffdd2.pth.tar',
        'display_name': 'RCAN (Original)',
        'color': 'blue'
    },
    'rcan_cbam_x4': {
        'arch': 'rcan_cbam_x4', 
        'weights_path': '/data/rcan/RCAN-PyTorch-master/results/RCAN_CBAM_x4-DIV2K/best.pth.tar',
        'display_name': 'RCAN + CBAM (Improved)',
        'color': 'red'
    }
}

# Dataset configuration
TEST_DATASET_CONFIG = {
    'base_dir': '/data/rcan/RCAN-PyTorch-master/data/DIV2K/RCAN',
    'gt_subdir': 'valid',
    'lr_subdir': 'LR',
    'upscale_factor': 4
}

# Test configuration
TEST_CONFIG = {
    'max_test_images': 40,  # Limit number of test images for faster execution
    'save_sample_images': 20,  # Number of sample images to save
    'batch_size': 1,
    'num_workers': 1,
    'only_test_y_channel': True
}

# Output configuration
OUTPUT_CONFIG = {
    'results_dir': './results/model_comparison_v2',
    'plots_subdir': 'plots',
    'images_subdir': 'images',
    'data_subdir': 'data'
}

# Visualization configuration
VIZ_CONFIG = {
    'figure_dpi': 300,
    'figure_format': 'png',
    'color_palette': 'husl',
    'style': 'seaborn-v0_8'
}

# Metrics to evaluate
METRICS_CONFIG = {
    'basic_metrics': ['psnr', 'ssim', 'mse', 'mae'],
    'advanced_metrics': ['edge_preservation', 'gradient_similarity', 'texture_similarity', 'sharpness_ratio'],
    'performance_metrics': ['inference_time']
}