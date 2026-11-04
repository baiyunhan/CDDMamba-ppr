from __future__ import annotations
from typing import Dict
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class ConvNormAct(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, stride: int=1) -> None:
        padding = kernel_size // 2
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False), nn.BatchNorm2d(out_channels), nn.GELU())

class DecoderBlock(nn.Module):

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(ConvNormAct(in_channels + skip_channels, out_channels), ConvNormAct(out_channels, out_channels))

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.projection(torch.cat((x, skip), dim=1))
