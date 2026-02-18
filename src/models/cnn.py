"""
src/models/cnn.py
─────────────────────────────────────────────────────────────────────────────
Baseline CNN for binary image classification (Cats vs Dogs).

Architecture:
    Conv(3→32) → BN → ReLU → MaxPool
    Conv(32→64) → BN → ReLU → MaxPool
    Conv(64→128) → BN → ReLU → MaxPool
    AdaptiveAvgPool → Flatten
    FC(128×7×7 → 256) → Dropout → ReLU → FC(256 → 1) → Sigmoid
"""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Simple 3-block CNN for binary classification.
    Input : (B, 3, 224, 224)
    Output: (B, 1)  — raw logit (use BCEWithLogitsLoss during training)
    """

    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()

        # ── Feature extractor ──────────────────────────────────────────────────
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 224 → 112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 112 → 56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 56 → 28
        )

        # Reduce spatial dims to 4×4 regardless of input size
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # ── Classifier ─────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),                       # single logit for binary cls
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x                                     # shape: (B, 1)


def build_model(architecture: str = "baseline_cnn", dropout: float = 0.5) -> nn.Module:
    """
    Factory function – returns the requested model.
    Supported: 'baseline_cnn'
    """
    if architecture == "baseline_cnn":
        return BaselineCNN(dropout=dropout)
    raise ValueError(f"Unknown architecture: {architecture!r}")