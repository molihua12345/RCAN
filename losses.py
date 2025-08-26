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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torch import Tensor
from typing import List

__all__ = [
    "PerceptualLoss",
    "CompositeLoss",
    "ESRGANDiscriminator",
]


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 features"""
    
    def __init__(self, feature_layers: List[int] = None, use_input_norm: bool = True):
        super(PerceptualLoss, self).__init__()
        if feature_layers is None:
            # Use conv5_4 features (before activation) as in ESRGAN
            feature_layers = [35]
        
        # Load pre-trained VGG19
        vgg = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:max(feature_layers) + 1])
        
        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.feature_layers = feature_layers
        self.use_input_norm = use_input_norm
        
        # ImageNet normalization
        if use_input_norm:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Calculate perceptual loss between x and y"""
        if self.use_input_norm:
            x = (x - self.mean) / self.std
            y = (y - self.mean) / self.std
        
        # Extract features
        x_features = []
        y_features = []
        
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            y = layer(y)
            if i in self.feature_layers:
                x_features.append(x)
                y_features.append(y)
        
        # Calculate L1 loss between features
        loss = 0
        for x_feat, y_feat in zip(x_features, y_features):
            loss += F.l1_loss(x_feat, y_feat)
        
        return loss / len(x_features)


class ESRGANDiscriminator(nn.Module):
    """ESRGAN-style discriminator for adversarial training"""
    
    def __init__(self, in_channels: int = 3, num_feat: int = 64):
        super(ESRGANDiscriminator, self).__init__()
        
        # Initial convolution
        self.conv0 = nn.Conv2d(in_channels, num_feat, 3, 1, 1)
        
        # Downsampling blocks
        self.conv1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_feat)
        
        self.conv2 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_feat * 2)
        
        self.conv3 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_feat * 2)
        
        self.conv4 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_feat * 4)
        
        self.conv5 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(num_feat * 4)
        
        self.conv6 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(num_feat * 8)
        
        self.conv7 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn7 = nn.BatchNorm2d(num_feat * 8)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat * 8, num_feat * 16, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feat * 16, 1, 1)
        )
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.lrelu(self.conv0(x))
        
        x = self.lrelu(self.bn1(self.conv1(x)))
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = self.lrelu(self.bn3(self.conv3(x)))
        x = self.lrelu(self.bn4(self.conv4(x)))
        x = self.lrelu(self.bn5(self.conv5(x)))
        x = self.lrelu(self.bn6(self.conv6(x)))
        x = self.lrelu(self.bn7(self.conv7(x)))
        
        x = self.classifier(x)
        
        return x.view(x.size(0), -1)


class CompositeLoss(nn.Module):
    """Composite loss combining L1, perceptual, and adversarial losses"""
    
    def __init__(self, 
                 l1_weight: float = 1.0,
                 perceptual_weight: float = 0.006,
                 adversarial_weight: float = 0.001,
                 feature_layers: List[int] = None):
        super(CompositeLoss, self).__init__()
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss(feature_layers)
        self.adversarial_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, 
                sr_images: Tensor, 
                hr_images: Tensor, 
                real_preds: Tensor = None, 
                fake_preds: Tensor = None) -> dict:
        """Calculate composite loss
        
        Args:
            sr_images: Super-resolution images
            hr_images: High-resolution ground truth images
            real_preds: Discriminator predictions for real images (optional)
            fake_preds: Discriminator predictions for fake images (optional)
        
        Returns:
            Dictionary containing individual losses and total loss
        """
        losses = {}
        
        # L1 loss
        losses['l1'] = self.l1_loss(sr_images, hr_images)
        
        # Perceptual loss
        losses['perceptual'] = self.perceptual_loss(sr_images, hr_images)
        
        # Adversarial loss (Relativistic GAN)
        if real_preds is not None and fake_preds is not None:
            # Relativistic discriminator loss for generator
            real_labels = torch.ones_like(fake_preds)
            fake_labels = torch.zeros_like(real_preds)
            
            loss_real = self.adversarial_loss(
                fake_preds - real_preds.mean(dim=0, keepdim=True), real_labels
            )
            loss_fake = self.adversarial_loss(
                real_preds - fake_preds.mean(dim=0, keepdim=True), fake_labels
            )
            
            losses['adversarial'] = (loss_real + loss_fake) / 2
        else:
            losses['adversarial'] = torch.tensor(0.0, device=sr_images.device)
        
        # Total loss
        losses['total'] = (
            self.l1_weight * losses['l1'] +
            self.perceptual_weight * losses['perceptual'] +
            self.adversarial_weight * losses['adversarial']
        )
        
        return losses


def relativistic_discriminator_loss(real_preds: Tensor, fake_preds: Tensor) -> Tensor:
    """Relativistic discriminator loss"""
    criterion = nn.BCEWithLogitsLoss()
    
    real_labels = torch.ones_like(real_preds)
    fake_labels = torch.zeros_like(fake_preds)
    
    loss_real = criterion(
        real_preds - fake_preds.mean(dim=0, keepdim=True), real_labels
    )
    loss_fake = criterion(
        fake_preds - real_preds.mean(dim=0, keepdim=True), fake_labels
    )
    
    return (loss_real + loss_fake) / 2