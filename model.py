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
import math

import torch
from torch import nn, Tensor

__all__ = [
    "RCAN", "RCAN_CBAM",
    "rcan_x2", "rcan_x3", "rcan_x4", "rcan_x8",
    "rcan_cbam_x2", "rcan_cbam_x3", "rcan_cbam_x4", "rcan_cbam_x8",
]


class RCAN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
            reduction: int,
            num_rcab: int,
            num_rg: int,
            upscale_factor: int,
            rgb_mean: tuple = None,
    ) -> None:
        super(RCAN, self).__init__()
        if rgb_mean is None:
            rgb_mean = [0.4488, 0.4371, 0.4040]

        # The first layer of convolutional layer
        self.conv1 = nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1))

        # Feature extraction backbone
        trunk = []
        for _ in range(num_rg):
            trunk.append(_ResidualGroup(channels, reduction, num_rcab))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))

        # Upsampling convolutional layer.
        upsampling = []
        if upscale_factor == 2 or upscale_factor == 4 or upscale_factor == 8:
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling.append(_UpsampleBlock(channels, 2))
        elif upscale_factor == 3:
            upsampling.append(_UpsampleBlock(channels, 3))
        self.upsampling = nn.Sequential(*upsampling)

        # Output layer.
        self.conv3 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

        self.register_buffer("mean", Tensor(rgb_mean).view(1, 3, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = x.sub_(self.mean).mul_(1.)

        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.upsampling(x)
        x = self.conv3(x)

        x = x.div_(1.).add_(self.mean)

        return x


class _ChannelAttentionCBAM(nn.Module):
    """Channel Attention Module in CBAM"""
    def __init__(self, channel: int, reduction: int = 16):
        super(_ChannelAttentionCBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class _SpatialAttentionCBAM(nn.Module):
    """Spatial Attention Module in CBAM"""
    def __init__(self, kernel_size: int = 7):
        super(_SpatialAttentionCBAM, self).__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_concat)
        return self.sigmoid(out)


class _CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channel: int, reduction: int = 16, kernel_size: int = 7):
        super(_CBAM, self).__init__()
        self.channel_attention = _ChannelAttentionCBAM(channel, reduction)
        self.spatial_attention = _SpatialAttentionCBAM(kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        # Channel attention
        x = x * self.channel_attention(x)
        # Spatial attention
        x = x * self.spatial_attention(x)
        return x


class _ChannelAttentionLayer(nn.Module):
    def __init__(self, channel: int, reduction: int):
        super(_ChannelAttentionLayer, self).__init__()
        self.channel_attention_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, (1, 1), (1, 1), (0, 0)),
            nn.ReLU(True),
            nn.Conv2d(channel // reduction, channel, (1, 1), (1, 1), (0, 0)),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.channel_attention_layer(x)

        out = torch.mul(out, x)

        return out


class _ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, channel: int, reduction: int):
        super(_ResidualChannelAttentionBlock, self).__init__()
        self.residual_channel_attention_block = nn.Sequential(
            nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1)),
            _ChannelAttentionLayer(channel, reduction),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.residual_channel_attention_block(x)

        out = torch.add(out, identity)

        return out


class _ResidualChannelAttentionBlockCBAM(nn.Module):
    """Residual Channel Attention Block with CBAM"""
    def __init__(self, channel: int, reduction: int = 16):
        super(_ResidualChannelAttentionBlockCBAM, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1))
        self.cbam = _CBAM(channel, reduction)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.cbam(out)
        out = torch.add(out, identity)
        return out


class _ResidualGroupCBAM(nn.Module):
    """Residual Group with CBAM"""
    def __init__(self, channel: int, reduction: int, num_rcab: int):
        super(_ResidualGroupCBAM, self).__init__()
        residual_group = []
        
        for _ in range(num_rcab):
            residual_group.append(_ResidualChannelAttentionBlockCBAM(channel, reduction))
        residual_group.append(nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1)))
        
        self.residual_group = nn.Sequential(*residual_group)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.residual_group(x)
        out = torch.add(out, identity)
        return out


class _ResidualGroup(nn.Module):
    def __init__(self, channel: int, reduction: int, num_rcab: int):
        super(_ResidualGroup, self).__init__()
        residual_group = []

        for _ in range(num_rcab):
            residual_group.append(_ResidualChannelAttentionBlock(channel, reduction))
        residual_group.append(nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1)))

        self.residual_group = nn.Sequential(*residual_group)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.residual_group(x)

        out = torch.add(out, identity)

        return out


class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample_block(x)

        return x


def rcan_x2(**kwargs) -> RCAN:
    model = RCAN(3, 3, 64, 16, 20, 10, 2, **kwargs)

    return model


def rcan_x3(**kwargs) -> RCAN:
    model = RCAN(3, 3, 64, 16, 20, 10, 3, **kwargs)

    return model


def rcan_x4(**kwargs) -> RCAN:
    model = RCAN(3, 3, 64, 16, 20, 10, 4, **kwargs)

    return model


def rcan_x8(**kwargs) -> RCAN:
    model = RCAN(3, 3, 64, 16, 20, 10, 8, **kwargs)

    return model


class RCAN_CBAM(nn.Module):
    """RCAN with CBAM attention mechanism"""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
            reduction: int,
            num_rcab: int,
            num_rg: int,
            upscale_factor: int,
            rgb_mean: tuple = None,
    ) -> None:
        super(RCAN_CBAM, self).__init__()
        if rgb_mean is None:
            rgb_mean = [0.4488, 0.4371, 0.4040]

        # The first layer of convolutional layer
        self.conv1 = nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1))

        # Feature extraction backbone with CBAM
        trunk = []
        for _ in range(num_rg):
            trunk.append(_ResidualGroupCBAM(channels, reduction, num_rcab))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))

        # Upsampling layers
        self.upsampling = _UpsampleBlock(channels, upscale_factor)

        # Output layer
        self.conv3 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

        # Initialize model weights
        self._initialize_weights()

        # Load the ImageNet dataset mean and std
        self.register_buffer("mean", torch.Tensor(rgb_mean).view(1, 3, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        # The images by subtracting the mean
        x = x.sub_(self.mean)

        # First layer
        conv1 = self.conv1(x)

        # Residual learning
        trunk = self.trunk(conv1)
        conv2 = self.conv2(trunk)
        x = torch.add(conv1, conv2)

        # Upsampling
        x = self.upsampling(x)

        # Output layer
        x = self.conv3(x)

        # The images by adding the mean
        x = x.add_(self.mean)

        x = torch.clamp_(x, 0.0, 1.0)

        return x

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


def rcan_cbam_x2(**kwargs) -> RCAN_CBAM:
    model = RCAN_CBAM(3, 3, 64, 16, 20, 10, 2, **kwargs)
    return model


def rcan_cbam_x3(**kwargs) -> RCAN_CBAM:
    model = RCAN_CBAM(3, 3, 64, 16, 20, 10, 3, **kwargs)
    return model


def rcan_cbam_x4(**kwargs) -> RCAN_CBAM:
    model = RCAN_CBAM(3, 3, 64, 16, 20, 10, 4, **kwargs)
    return model


def rcan_cbam_x8(**kwargs) -> RCAN_CBAM:
    model = RCAN_CBAM(3, 3, 64, 16, 20, 10, 8, **kwargs)
    return model
