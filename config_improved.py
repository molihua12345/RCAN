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
Improved RCAN Configuration File

This configuration file demonstrates how to set up training with:
1. RCAN_CBAM model (RCAN with CBAM attention)
2. Composite loss function (L1 + Perceptual + Adversarial)

To use this configuration:
1. Copy the relevant settings to your main config.py file, or
2. Import this file and use the settings directly

Example usage:
    # Option 1: Copy settings to config.py
    # Copy the relevant variables below to your config.py
    
    # Option 2: Import and use
    import config_improved as config
    # Then use config.model_arch_name, etc.
"""

import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True

# Only test the Y channel of the image
only_test_y_channel = True

# Model architecture name
# Options: "rcan_x4" (original), "rcan_cbam_x4" (with CBAM attention)
model_arch_name = "rcan_cbam_x4"  # Use CBAM version for improved performance

# Model magnification
upscale_factor = 4

# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "RCAN_CBAM_x4-DIV2K-Composite"

# ============================================================================
# Training Configuration
# ============================================================================
if mode == "train":
    # Dataset paths
    train_gt_images_dir = f"./data/DIV2K/RCAN/train"
    test_gt_images_dir = f"./data/Set5/GTmod12"
    test_lr_images_dir = f"./data/Set5/LRbicx{upscale_factor}"

    # Training parameters
    train_gt_image_size = int(upscale_factor * 48)
    batch_size = 16  # Adjust based on GPU memory
    num_workers = 4

    # Model loading paths
    # Load the address of the pretrained model (optional)
    pretrained_model_weights_path = f"./results/pretrained_models/RCAN_x4-DIV2K-2dfffdd2.pth.tar"
    
    # Incremental training and migration training (optional)
    resume_model_weights_path = f""

    # Training epochs
    epochs = 1000

    # ========================================================================
    # Composite Loss Configuration
    # ========================================================================
    # Enable composite loss (L1 + Perceptual + Adversarial)
    use_composite_loss = True
    
    # Loss weights (adjust these based on your needs)
    l1_weight = 1.0              # Weight for L1 loss
    perceptual_weight = 0.006    # Weight for perceptual loss (VGG features)
    adversarial_weight = 0.001   # Weight for adversarial loss
    
    # Discriminator learning rate (for adversarial training)
    discriminator_lr = 1e-4

    # ========================================================================
    # Optimizer Configuration
    # ========================================================================
    # Generator optimizer parameters
    model_lr = 1e-4              # Learning rate (use smaller for fine-tuning)
    model_betas = (0.9, 0.99)    # Adam beta parameters
    model_eps = 1e-4             # Adam epsilon (keep no nan)
    model_weight_decay = 0.0     # Weight decay

    # EMA (Exponential Moving Average) parameter
    model_ema_decay = 0.999

    # Learning rate scheduler parameters
    lr_scheduler_step_size = epochs // 5  # Step size for StepLR
    lr_scheduler_gamma = 0.5               # Gamma for StepLR

    # Training output frequency
    train_print_frequency = 100  # Print training info every N batches
    test_print_frequency = 1     # Print test info every N epochs

# ============================================================================
# Testing Configuration
# ============================================================================
if mode == "test":
    # Test dataset paths
    test_gt_images_dir = f"./data/Set5/GTmod12"
    test_sr_images_dir = f"./results/test/{exp_name}"
    test_lr_images_dir = f"./data/Set5/LRbicx{upscale_factor}"

    # Model weights path for testing
    # Use the best model from improved training
    model_weights_path = f"./results/{exp_name}/best.pth.tar"

# ============================================================================
# Alternative Configurations
# ============================================================================

# Configuration for original RCAN (without improvements)
class OriginalRCANConfig:
    """Configuration for original RCAN model"""
    model_arch_name = "rcan_x4"
    use_composite_loss = False
    exp_name = "RCAN_x4-DIV2K-Original"
    model_lr = 1e-4

# Configuration for CBAM only (without composite loss)
class CBAMOnlyConfig:
    """Configuration for RCAN with CBAM but L1 loss only"""
    model_arch_name = "rcan_cbam_x4"
    use_composite_loss = False
    exp_name = "RCAN_CBAM_x4-DIV2K-L1Only"
    model_lr = 1e-4

# Configuration for fine-tuning with lower learning rate
class FineTuningConfig:
    """Configuration for fine-tuning with lower learning rate"""
    model_arch_name = "rcan_cbam_x4"
    use_composite_loss = True
    exp_name = "RCAN_CBAM_x4-DIV2K-FineTune"
    model_lr = 1e-5  # Lower learning rate for fine-tuning
    l1_weight = 1.0
    perceptual_weight = 0.01     # Higher perceptual weight
    adversarial_weight = 0.005   # Higher adversarial weight

# ============================================================================
# Usage Examples
# ============================================================================
"""
Example 1: Use improved RCAN with CBAM and composite loss
---------------------------------------------------------
# In your training script:
import config_improved as config
# config.model_arch_name will be "rcan_cbam_x4"
# config.use_composite_loss will be True

Example 2: Use only CBAM without composite loss
-----------------------------------------------
# In your training script:
import config_improved as config
from config_improved import CBAMOnlyConfig

# Override specific settings
config.model_arch_name = CBAMOnlyConfig.model_arch_name
config.use_composite_loss = CBAMOnlyConfig.use_composite_loss
config.exp_name = CBAMOnlyConfig.exp_name

Example 3: Fine-tuning configuration
-----------------------------------
# In your training script:
import config_improved as config
from config_improved import FineTuningConfig

# Override for fine-tuning
config.model_lr = FineTuningConfig.model_lr
config.perceptual_weight = FineTuningConfig.perceptual_weight
config.adversarial_weight = FineTuningConfig.adversarial_weight
config.exp_name = FineTuningConfig.exp_name
"""