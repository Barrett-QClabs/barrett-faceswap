"""Scaffold for a commercially licensable swapper. Nothing here trains or downloads."""

from barrett_faceswap.train.metrics import identity_cosine, ssim
from barrett_faceswap.train.scaffold import export_contract, training_plan, validate_dataset

__all__ = ["export_contract", "identity_cosine", "ssim", "training_plan", "validate_dataset"]
