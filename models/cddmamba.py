from __future__ import annotations
from .layers import ConvNormAct, DecoderBlock
from typing import Dict
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .cdfm import CDFM
from .dp_mpc import DPMPC
from .dgr_csmamba import DGRCSMamba

class CDDMamba(nn.Module):
    """RegNet-Y + CDFM + DP-MPC + DGR-CSMamba + detail recovery."""

    def __init__(self, *, num_classes: int, fusion_dim: int, polynomial_order: int, cdfm_clear_ratio: float, cdfm_lambda_clear: float, cdfm_lambda_ambiguous: float, dgr_candidate_size: int, dgr_state_dim: int, dgr_scan_mode: str, dropout: float, backbone, cdfm_rk_steps, scan_alpha, scan_beta, scan_eta) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        c1, c2, c3, c4 = self.backbone.feature_info.channels()
        self.cdfm = CDFM(channels=c4, hidden_channels=fusion_dim, clear_ratio=cdfm_clear_ratio, lambda_clear=cdfm_lambda_clear, lambda_ambiguous=cdfm_lambda_ambiguous, rk_steps=cdfm_rk_steps)
        self.dp_mpc1 = DPMPC(c1, kernel_size=5, polynomial_order=polynomial_order)
        self.dp_mpc2 = DPMPC(c2, kernel_size=7, polynomial_order=polynomial_order)
        self.dp_mpc3 = DPMPC(c3, kernel_size=9, polynomial_order=polynomial_order)
        self.local_projection1 = nn.Conv2d(c1, fusion_dim, 1, bias=False)
        self.local_projection2 = nn.Conv2d(c2, fusion_dim, 1, bias=False)
        self.local_projection3 = nn.Conv2d(c3, fusion_dim, 1, bias=False)
        self.local_fusion = nn.Sequential(ConvNormAct(fusion_dim * 3, fusion_dim, 1), ConvNormAct(fusion_dim, fusion_dim))
        self.dgr_csmamba = DGRCSMamba(global_channels=c4, local_channels=fusion_dim, dim=fusion_dim, candidate_size=dgr_candidate_size, d_state=dgr_state_dim, dropout=dropout, scan_mode=dgr_scan_mode, alpha=scan_alpha, beta=scan_beta, eta=scan_eta)
        self.decoder2 = DecoderBlock(fusion_dim, c2, 128)
        self.decoder1 = DecoderBlock(128, c1, 64)
        self.detail_stem = nn.Sequential(ConvNormAct(3, 32, 3, stride=2), ConvNormAct(32, 32))
        self.detail_decoder = DecoderBlock(64, 32, 64)
        self.segmentation_head = nn.Sequential(ConvNormAct(64, 64), nn.Dropout2d(dropout), nn.Conv2d(64, num_classes, 1))

    def _local_pyramid(self, p1: Tensor, p2: Tensor, p3: Tensor) -> Tensor:
        target_size = p3.shape[-2:]
        p1_down = self.local_projection1(F.adaptive_avg_pool2d(p1, target_size))
        p2_down = self.local_projection2(F.adaptive_avg_pool2d(p2, target_size))
        p3_projected = self.local_projection3(p3)
        return self.local_fusion(torch.cat((p1_down, p2_down, p3_projected), dim=1))

    def forward(self, x: Tensor) -> Tensor:
        input_size = x.shape[-2:]
        detail = self.detail_stem(x)
        e1, e2, e3, e4 = self.backbone(x)
        global_feature = self.cdfm(e4)
        p1 = self.dp_mpc1(e1)
        p2 = self.dp_mpc2(e2)
        p3 = self.dp_mpc3(e3)
        local_feature = self._local_pyramid(p1, p2, p3)
        fused = self.dgr_csmamba(global_feature, local_feature)
        decoded2 = self.decoder2(fused, p2)
        decoded1 = self.decoder1(decoded2, p1)
        detailed = self.detail_decoder(decoded1, detail)
        logits = self.segmentation_head(detailed)
        logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        return logits
