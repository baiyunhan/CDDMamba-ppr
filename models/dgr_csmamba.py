from __future__ import annotations
from functools import lru_cache
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from mamba_ssm import Mamba

def _rotate(n: int, x: int, y: int, rx: int, ry: int) -> tuple[int, int]:
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = (y, x)
    return (x, y)

def _hilbert_xy(side: int, distance: int) -> tuple[int, int]:
    x = y = 0
    scale = 1
    value = distance
    while scale < side:
        rx = 1 & value // 2
        ry = 1 & (value ^ rx)
        x, y = _rotate(scale, x, y, rx, ry)
        x += scale * rx
        y += scale * ry
        value //= 4
        scale *= 2
    return (x, y)

@lru_cache(maxsize=32)
def hilbert_order(height: int, width: int) -> Tensor:
    """Hilbert order filtered to an arbitrary rectangular feature map."""
    side = 1
    while side < max(height, width):
        side *= 2
    order = []
    for distance in range(side * side):
        x, y = _hilbert_xy(side, distance)
        if y < height and x < width:
            order.append(y * width + x)
    return torch.tensor(order, dtype=torch.long)

def _batched_gather(sequence: Tensor, indices: Tensor) -> Tensor:
    return torch.gather(sequence, 1, indices.unsqueeze(-1).expand(-1, -1, sequence.shape[-1]))

class DifferenceGuidedScan(nn.Module):
    """Vectorised local-candidate realisation of the paper's Hilbert-style scan.

    The base Hilbert curve preserves spatial adjacency. Within each bounded local
    candidate window, tokens are selected by difference priority plus a continuity
    prior. This avoids an O(N^2) Python traversal while retaining the stated rule.
    """

    def __init__(self, *, candidate_size: int, alpha: float, beta: float, eta: float) -> None:
        super().__init__()
        if candidate_size < 2:
            raise ValueError('candidate_size must be at least 2')
        self.candidate_size = candidate_size
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.eta = nn.Parameter(torch.tensor(float(eta)))

    def forward(self, difference: Tensor, height: int, width: int) -> Tensor:
        batch, token_count = difference.shape
        base_order = hilbert_order(height, width).to(difference.device)
        if base_order.numel() != token_count:
            raise RuntimeError('Hilbert order and token count do not match')
        difference_h = torch.gather(difference, 1, base_order[None, :].expand(batch, -1))
        window = self.candidate_size
        padding = -token_count % window
        if padding:
            difference_h = F.pad(difference_h, (0, padding), value=-10000.0)
            padded_order = F.pad(base_order, (0, padding), value=-1)
        else:
            padded_order = base_order
        difference_h = difference_h.view(batch, -1, window)
        position = torch.linspace(0.0, 1.0, window, device=difference.device, dtype=difference.dtype)
        score = self.alpha.abs() * difference_h - self.beta.abs() * position[None, None, :] + self.eta.abs() * (1.0 - position[None, None, :])
        local_choice = score.argsort(dim=-1, descending=True)
        order_windows = padded_order.view(1, -1, window).expand(batch, -1, -1)
        dynamic_order = torch.gather(order_windows, 2, local_choice).flatten(1)
        if padding:
            dynamic_order = dynamic_order[:, :token_count]
        return dynamic_order

class CoupledStateBlock(nn.Module):

    def __init__(self, dim: int, d_state: int=16, d_conv: int=4, expand: int=2, dropout: float=0.0) -> None:
        super().__init__()
        self.global_norm = nn.LayerNorm(dim)
        self.local_norm = nn.LayerNorm(dim)
        self.global_cross = nn.Linear(dim, dim, bias=False)
        self.local_cross = nn.Linear(dim, dim, bias=False)
        self.gamma = nn.Sequential(nn.Linear(1, max(8, dim // 8)), nn.SiLU(), nn.Linear(max(8, dim // 8), 1), nn.Sigmoid())
        self.interleave_mixer = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.global_mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.local_mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, global_tokens: Tensor, local_tokens: Tensor, difference: Tensor) -> tuple[Tensor, Tensor]:
        batch, token_count, channels = global_tokens.shape
        interleaved = torch.stack((global_tokens, local_tokens), dim=2).reshape(batch, token_count * 2, channels)
        interleaved = interleaved + self.interleave_mixer(interleaved.transpose(1, 2)).transpose(1, 2)
        paired = interleaved.view(batch, token_count, 2, channels)
        global_tokens, local_tokens = (paired[:, :, 0], paired[:, :, 1])
        coupling = self.gamma(difference.unsqueeze(-1))
        global_input = self.global_norm(global_tokens + coupling * self.global_cross(local_tokens))
        local_input = self.local_norm(local_tokens + coupling * self.local_cross(global_tokens))
        global_output = global_tokens + self.dropout(self.global_mamba(global_input))
        local_output = local_tokens + self.dropout(self.local_mamba(local_input))
        return (global_output, local_output)

class DGRCSMamba(nn.Module):

    def __init__(self, *, global_channels: int, local_channels: int, dim: int, candidate_size: int, d_state: int, dropout: float, scan_mode: str, alpha: float, beta: float, eta: float) -> None:
        super().__init__()
        if scan_mode not in {'difference_guided_hilbert', 'hilbert', 'raster'}:
            raise ValueError(f'Unsupported scan_mode: {scan_mode}')
        self.dim = dim
        self.scan_mode = scan_mode
        self.global_projection = nn.Conv2d(global_channels, dim, 1, bias=False)
        self.local_projection = nn.Conv2d(local_channels, dim, 1, bias=False)
        self.global_token_norm = nn.LayerNorm(dim)
        self.local_token_norm = nn.LayerNorm(dim)
        self.scan = DifferenceGuidedScan(candidate_size=candidate_size, alpha=alpha, beta=beta, eta=eta) if scan_mode == 'difference_guided_hilbert' else None
        self.coupled_state = CoupledStateBlock(dim=dim, d_state=d_state, dropout=dropout)
        self.fusion = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, global_feature: Tensor, local_feature: Tensor) -> Tensor:
        if global_feature.shape[-2:] != local_feature.shape[-2:]:
            global_feature = F.interpolate(global_feature, size=local_feature.shape[-2:], mode='bilinear', align_corners=False)
        global_feature = self.global_projection(global_feature)
        local_feature = self.local_projection(local_feature)
        batch, _, height, width = global_feature.shape
        global_tokens = global_feature.flatten(2).transpose(1, 2)
        local_tokens = local_feature.flatten(2).transpose(1, 2)
        global_normalised = self.global_token_norm(global_tokens)
        local_normalised = self.local_token_norm(local_tokens)
        difference = (global_normalised - local_normalised).square().mean(dim=-1).sqrt()
        difference = (difference - difference.amin(dim=1, keepdim=True)) / (difference.amax(dim=1, keepdim=True) - difference.amin(dim=1, keepdim=True)).clamp_min(1e-06)
        if self.scan_mode == 'difference_guided_hilbert':
            scan_indices = self.scan(difference.detach(), height, width)
        elif self.scan_mode == 'hilbert':
            scan_indices = hilbert_order(height, width).to(difference.device)
            scan_indices = scan_indices[None, :].expand(batch, -1)
        else:
            scan_indices = torch.arange(height * width, device=difference.device, dtype=torch.long)[None, :].expand(batch, -1)
        scanned_global = _batched_gather(global_tokens, scan_indices)
        scanned_local = _batched_gather(local_tokens, scan_indices)
        scanned_difference = torch.gather(difference, 1, scan_indices)
        global_state, local_state = self.coupled_state(scanned_global, scanned_local, scanned_difference)
        fused = self.fusion(torch.cat((global_state, local_state, global_state - local_state), dim=-1))
        fused = self.output_norm(fused + 0.5 * (global_state + local_state))
        recovered = torch.zeros_like(fused)
        recovered.scatter_(1, scan_indices.unsqueeze(-1).expand_as(fused), fused)
        return recovered.transpose(1, 2).reshape(batch, self.dim, height, width)
