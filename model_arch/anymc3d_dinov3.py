"""Patient-level AnyMC3D classifier using official Transformers DINOv3."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from importlib.metadata import version

import torch
from packaging.version import Version
from torch import nn

from .lora_dinov3 import (
    LoRAInjectionReport,
    assert_only_lora_trainable,
    inject_lora_dinov3,
)

LOGGER = logging.getLogger(__name__)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MIN_TRANSFORMERS_VERSION = Version("4.56.0")


@dataclass
class AnyMC3DOutput:
    """One binary prediction and slice-attention distribution per patient."""

    logits: torch.Tensor
    attention_weights: torch.Tensor
    slice_mask: torch.Tensor


class MaskedQueryAttentionPool(nn.Module):
    """Permutation-invariant query pooling with strict slice masking."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.task_query = nn.Parameter(torch.empty(embedding_dim))
        nn.init.trunc_normal_(self.task_query, std=0.02)

    def forward(
        self, embeddings: torch.Tensor, slice_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if embeddings.ndim != 3:
            raise ValueError(
                f"Expected embeddings [B,S,D], got {tuple(embeddings.shape)}"
            )
        if slice_mask.shape != embeddings.shape[:2]:
            raise ValueError(
                f"slice_mask {tuple(slice_mask.shape)} does not match "
                f"embeddings {tuple(embeddings.shape[:2])}"
            )
        slice_mask = slice_mask.bool()
        if not torch.all(slice_mask.any(dim=1)):
            raise ValueError("Every patient must contain at least one real slice")
        scores = torch.einsum(
            "bsd,d->bs", embeddings, self.task_query
        ) / math.sqrt(self.embedding_dim)
        scores = scores.masked_fill(~slice_mask, float("-inf"))
        attention = torch.softmax(scores.float(), dim=1).to(embeddings.dtype)
        attention = attention.masked_fill(~slice_mask, 0.0)
        volume_embedding = torch.einsum(
            "bs,bsd->bd", attention, embeddings
        )
        return volume_embedding, attention


def _transformers_version() -> Version:
    return Version(version("transformers"))


def load_dinov3_backbone(backbone_name: str) -> nn.Module:
    """Load only the requested official HF checkpoint with explicit failures."""

    installed = _transformers_version()
    if installed < MIN_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"Transformers >= {MIN_TRANSFORMERS_VERSION} is required for "
            f"DINOv3, found {installed}"
        )
    try:
        from transformers import AutoModel, DINOv3ViTModel
    except ImportError as exc:
        raise RuntimeError(
            "This Transformers installation does not expose DINOv3ViTModel. "
            "Install a compatible release >=4.56.0."
        ) from exc
    try:
        backbone = AutoModel.from_pretrained(backbone_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the requested DINOv3 checkpoint '{backbone_name}'. "
            "The Meta DINOv3 repositories are gated: accept the license on "
            "Hugging Face and authenticate with `hf auth login`. The model "
            f"will not be replaced by a fallback. Original error: {exc}"
        ) from exc
    if not isinstance(backbone, DINOv3ViTModel):
        raise TypeError(
            f"Checkpoint '{backbone_name}' resolved to "
            f"{type(backbone).__name__}, not DINOv3ViTModel"
        )
    return backbone


class AnyMC3DDINOv3(nn.Module):
    """DINOv3+LoRA slice encoder and one-logit patient classifier."""

    def __init__(
        self,
        *,
        backbone_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        slice_chunk_size: int = 8,
        gradient_checkpointing: bool = False,
        head_dropout: float = 0.0,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if slice_chunk_size <= 0:
            raise ValueError("slice_chunk_size must be positive")
        if not 0.0 <= float(head_dropout) < 1.0:
            raise ValueError("head_dropout must be in [0, 1)")
        self.backbone_name = backbone_name
        self.slice_chunk_size = int(slice_chunk_size)
        self.backbone = (
            backbone if backbone is not None else load_dinov3_backbone(backbone_name)
        )
        if getattr(self.backbone.config, "model_type", None) != "dinov3_vit":
            raise TypeError(
                f"Expected a DINOv3 ViT backbone, got model_type="
                f"{getattr(self.backbone.config, 'model_type', None)!r}"
            )
        if gradient_checkpointing:
            if not hasattr(self.backbone, "gradient_checkpointing_enable"):
                raise RuntimeError(
                    "This DINOv3 implementation does not support gradient checkpointing"
                )
            self.backbone.gradient_checkpointing_enable()

        self.hidden_size = int(self.backbone.config.hidden_size)
        patch_size = self.backbone.config.patch_size
        if isinstance(patch_size, (tuple, list)):
            if len(patch_size) != 2 or patch_size[0] != patch_size[1]:
                raise ValueError(f"Unsupported DINOv3 patch size: {patch_size}")
            patch_size = patch_size[0]
        self.patch_size = int(patch_size)
        self.num_register_tokens = int(
            getattr(self.backbone.config, "num_register_tokens", 0)
        )
        self.lora_report: LoRAInjectionReport = inject_lora_dinov3(
            self.backbone, rank=lora_rank, alpha=lora_alpha
        )
        assert_only_lora_trainable(self.backbone)
        self.attention_pool = MaskedQueryAttentionPool(self.hidden_size)
        self.classification_dropout = nn.Dropout(float(head_dropout))
        self.classification_head = nn.Linear(self.hidden_size, 1)
        self.register_buffer(
            "imagenet_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )
        self.last_num_encoded_slices = 0

        report = self.parameter_report()
        if report["trainable_parameters"] > 10_000_000:
            LOGGER.warning(
                "Unexpectedly high trainable parameter count: %s",
                report["trainable_parameters"],
            )

    def parameter_report(self) -> dict[str, float | int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "trainable_percent": 100.0 * trainable / total,
        }

    def trainable_parameter_names(self) -> list[str]:
        return [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]

    def _validate_canvas(self, height: int, width: int) -> None:
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Canvas {(height, width)} must be divisible by DINOv3 patch "
                f"size {self.patch_size}"
            )

    def _prepare_slices(self, slices: torch.Tensor) -> torch.Tensor:
        if slices.ndim != 4 or slices.shape[1] != 1:
            raise ValueError(
                f"Expected selected slices [N,1,H,W], got {tuple(slices.shape)}"
            )
        slices = slices.repeat(1, 3, 1, 1)
        # No resize, crop, processor, or hidden range conversion is performed.
        return (slices - self.imagenet_mean) / self.imagenet_std

    def _encode_chunk(self, slices: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=slices)
        if not hasattr(outputs, "last_hidden_state"):
            raise RuntimeError("DINOv3 output has no last_hidden_state field")
        hidden = outputs.last_hidden_state
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise RuntimeError(
                f"Unexpected DINOv3 last_hidden_state: {tuple(hidden.shape)}"
            )
        # Transformers order: CLS, register tokens, then patch tokens.
        return hidden[:, 0, :]

    def encode_valid_slices(
        self,
        volume: torch.Tensor,
        slice_mask: torch.Tensor,
        *,
        slice_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Encode real slices only and reconstruct zero-padded [B,S,D]."""

        if volume.ndim != 5:
            raise ValueError(
                f"Expected volume [B,S,1,H,W], got {tuple(volume.shape)}"
            )
        batch, slices, channels, height, width = volume.shape
        if channels != 1:
            raise ValueError(f"Expected one CT channel, got {channels}")
        if slice_mask.shape != (batch, slices):
            raise ValueError(
                f"slice_mask {tuple(slice_mask.shape)} != {(batch, slices)}"
            )
        self._validate_canvas(height, width)
        slice_mask = slice_mask.bool()
        if not torch.all(slice_mask.any(dim=1)):
            raise ValueError("Every patient must contain at least one real slice")
        flat_mask = slice_mask.reshape(-1)
        flat_volume = volume.reshape(batch * slices, 1, height, width)
        selected = self._prepare_slices(flat_volume[flat_mask])
        chunk_size = int(slice_chunk_size or self.slice_chunk_size)
        if chunk_size <= 0:
            raise ValueError("slice_chunk_size must be positive")
        valid_embeddings = torch.cat(
            [
                self._encode_chunk(chunk)
                for chunk in selected.split(chunk_size, dim=0)
            ],
            dim=0,
        )
        embeddings = valid_embeddings.new_zeros(
            (batch * slices, self.hidden_size)
        )
        embeddings = embeddings.index_copy(
            0, flat_mask.nonzero(as_tuple=False).squeeze(1), valid_embeddings
        )
        self.last_num_encoded_slices = int(flat_mask.sum().item())
        return embeddings.view(batch, slices, self.hidden_size)

    def forward(
        self,
        volume: torch.Tensor,
        slice_mask: torch.Tensor,
        *,
        slice_chunk_size: int | None = None,
    ) -> AnyMC3DOutput:
        embeddings = self.encode_valid_slices(
            volume,
            slice_mask,
            slice_chunk_size=slice_chunk_size,
        )
        patient_embedding, attention = self.attention_pool(
            embeddings, slice_mask
        )
        logits = self.classification_head(
            self.classification_dropout(patient_embedding)
        )
        return AnyMC3DOutput(
            logits=logits,
            attention_weights=attention,
            slice_mask=slice_mask.bool(),
        )
