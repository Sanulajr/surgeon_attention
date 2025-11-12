from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ------------------------ CBAM blocks ------------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(1, in_planes // ratio)
        self.fc1 = nn.Conv2d(in_planes, hidden, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


# ------------------------ Backbone ------------------------

class ResNetBackbone(nn.Module):
    def __init__(self, name: str = "resnet18", pretrained: bool = True, att_ratio: int = 16):
        super().__init__()
        if name == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            feat_dim = 512
        elif name == "resnet34":
            net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            feat_dim = 512
        elif name == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone {name}")

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.cbam = CBAM(feat_dim, ratio=att_ratio)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = feat_dim

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        x = self.pool(x).flatten(1)
        return x


# ------------------------ Anti-collapse Fusion ------------------------

class MultiViewFusion(nn.Module):
    """
    Fuses per-view features (B, V, D) -> (B, D) and returns (fused, att).
    Anti-collapse tricks:
      - temperature softmax
      - train-only small Gumbel noise on logits
      - min-prob clamp + renorm
      - scorer dropout
      - optional warmup to mean fusion
    """
    def __init__(
        self,
        feat_dim: int,
        mode: str = "attention",
        temperature: float = 0.7,
        min_prob: float = 1e-4,
        scorer_dropout: float = 0.1,
    ):
        super().__init__()
        self.mode = mode
        self.temperature = max(1e-3, float(temperature))
        self.min_prob = float(min_prob)

        if mode == "attention":
            hidden = max(8, feat_dim // 2)
            self.score = nn.Sequential(
                nn.LayerNorm(feat_dim),           # normalize features across D
                nn.Dropout(p=scorer_dropout),     # regularize scorer
                nn.Linear(feat_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 1)
            )

    @staticmethod
    def _sample_gumbel_like(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
        # Gumbel(0,1) noise for tiny exploration; numerically stable.
        u = torch.rand_like(x)
        return -torch.log(-torch.log(u + eps) + eps)

    def forward(self, feats: torch.Tensor, force_mean_fusion: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        feats: (B, V, D)
        returns:
          fused: (B, D)
          att:   (B, V) or None
        """
        if feats.dim() != 3:
            raise ValueError("feats must be (B, V, D)")

        if self.mode in ("mean", "max") or force_mean_fusion:
            if self.mode == "max" and not force_mean_fusion:
                return feats.max(dim=1).values, None
            return feats.mean(dim=1), None

        # attention
        B, V, D = feats.shape
        logits = self.score(feats).view(B, V, 1)  # (B, V, 1)

        # Tiny exploration noise during training so one view can’t win too early
        if self.training:
            g = self._sample_gumbel_like(logits) * 0.03  # 0.03 is safe
            logits = logits + g

        att = torch.softmax(logits / self.temperature, dim=1)        # (B, V, 1)
        att = torch.clamp(att, min=self.min_prob)                    # floor
        att = att / att.sum(dim=1, keepdim=True)                     # renorm
        fused = (feats * att).sum(dim=1)                             # (B, D)
        return fused, att.squeeze(-1)


# ------------------------ Full Model ------------------------

class AttentionPoseGazeNet(nn.Module):
    """
    CNN backbone + CBAM + multiview attention fusion.
    Outputs head pose regression (yaw, pitch) and gaze class logits.
    """
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        att_ratio: int = 16,
        gaze_classes: int = 5,
        fusion: str = "attention",
        dropout: float = 0.2,
        fusion_temperature: float = 0.7,
        fusion_min_prob: float = 1e-4,
        fusion_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = ResNetBackbone(name=backbone, pretrained=pretrained, att_ratio=att_ratio)
        self.fusion = MultiViewFusion(
            self.backbone.out_dim,
            mode=fusion,
            temperature=fusion_temperature,
            min_prob=fusion_min_prob,
            scorer_dropout=fusion_dropout,
        )
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
        D = self.backbone.out_dim
        self.pose_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Linear(D // 2, 2)  # yaw, pitch
        )
        self.gaze_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Linear(D // 2, gaze_classes)
        )

    def encode_views(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, V, 3, H, W) -> feats: (B, V, D)"""
        B, V = x.shape[:2]
        x = x.view(B * V, *x.shape[2:])
        feats = self.backbone(x)             # (B*V, D)
        feats = feats.view(B, V, -1)         # (B, V, D)
        return feats

    def forward(self, x: torch.Tensor, force_mean_fusion: bool = False):
        feats = self.encode_views(x)                          # (B, V, D)
        fused, att = self.fusion(feats, force_mean_fusion)    # (B, D), (B, V) or None
        pose = self.pose_head(fused)
        gaze_logits = self.gaze_head(fused)
        return pose, gaze_logits, att
