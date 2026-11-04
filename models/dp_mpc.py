"""Dual-Path Multi-Axis Polynomial Convolution (DP-MPC)."""
from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class ChannelLayerNorm2d(nn.Module):
    """LayerNorm over channels independently at every spatial position."""

    def __init__(self, channels: int, eps: float=1e-06) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]

def _legendre_basis(order: int, kernel_size: int) -> Tensor:
    radius = kernel_size // 2
    coordinates = torch.linspace(-1.0, 1.0, kernel_size)
    polynomials = [torch.ones_like(coordinates)]
    if order >= 1:
        polynomials.append(coordinates)
    for degree in range(2, order + 1):
        polynomial = ((2 * degree - 1) * coordinates * polynomials[-1] - (degree - 1) * polynomials[-2]) / degree
        polynomials.append(polynomial)
    basis = torch.stack(polynomials, dim=0)
    basis = basis / basis.norm(dim=1, keepdim=True).clamp_min(1e-06)
    if radius == 0:
        raise ValueError('kernel_size must be at least 3')
    return basis

class PolynomialAxisConvolution(nn.Module):
    """Depthwise H/W/two-diagonal strip convolutions parameterised by Legendre bases."""

    def __init__(self, channels: int, kernel_size: int, order: int=3) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError('kernel_size must be an odd integer >= 3')
        if order < 0:
            raise ValueError('order must be non-negative')
        self.channels = channels
        self.kernel_size = kernel_size
        self.order = order
        self.padding = kernel_size // 2
        self.register_buffer('basis', _legendre_basis(order, kernel_size), persistent=False)
        self.coefficients = nn.Parameter(torch.empty(4, channels, order + 1))
        nn.init.normal_(self.coefficients, std=0.02)

    def _one_dimensional_kernels(self) -> Tensor:
        return torch.einsum('acr,rk->ack', self.coefficients, self.basis)

    def _two_dimensional_kernels(self) -> tuple[Tensor, Tensor, Tensor]:
        kernels_1d = self._one_dimensional_kernels()
        k = self.kernel_size
        center = self.padding
        horizontal = kernels_1d.new_zeros(self.channels, 1, k, k)
        vertical = kernels_1d.new_zeros(self.channels, 1, k, k)
        diagonal = kernels_1d.new_zeros(self.channels, 1, k, k)
        horizontal[:, 0, center, :] = kernels_1d[0]
        vertical[:, 0, :, center] = kernels_1d[1]
        indices = torch.arange(k, device=kernels_1d.device)
        diagonal[:, 0, indices, indices] = kernels_1d[2]
        diagonal[:, 0, indices, k - 1 - indices] += kernels_1d[3]
        return (horizontal, vertical, diagonal)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        horizontal, vertical, diagonal = self._two_dimensional_kernels()
        kwargs = dict(padding=self.padding, groups=self.channels)
        return (F.conv2d(x, horizontal, **kwargs), F.conv2d(x, vertical, **kwargs), F.conv2d(x, diagonal, **kwargs))

class DPMPC(nn.Module):
    """Local DoubleConv and multi-axis polynomial strip paths."""

    def __init__(self, channels: int, kernel_size: int, polynomial_order: int=3, dropout: float=0.0) -> None:
        super().__init__()
        self.channels = channels
        self.local_path = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.GELU(), nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.GELU())
        self.axis_path = PolynomialAxisConvolution(channels, kernel_size=kernel_size, order=polynomial_order)
        self.output_projection = nn.Sequential(nn.Conv2d(channels * 4, channels, 1, bias=False), nn.Dropout2d(dropout) if dropout > 0 else nn.Identity())
        self.norm = ChannelLayerNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f'DPMPC expected [B, {self.channels}, H, W], got {tuple(x.shape)}')
        local = self.local_path(x)
        horizontal, vertical, diagonal = self.axis_path(x)
        projected = self.output_projection(torch.cat((local, horizontal, vertical, diagonal), dim=1))
        return self.norm(x + projected)
