"""Explicit LoRA adapters for the Hugging Face DINOv3 ViT implementation.

Transformers exposes separate ``q_proj``, ``k_proj`` and ``v_proj`` modules.
To match a fused-QKV rank-r update, the three projections within a block
share the same low-rank input matrix A and learn independent output matrices
Bq/Bk/Bv. This is algebraically equivalent to LoRA on a fused QKV matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class LinearLoRA(nn.Module):
    """Frozen linear layer plus a trainable low-rank residual."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        shared_a: nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        if shared_a is None:
            self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            if tuple(shared_a.shape) != (rank, base.in_features):
                raise ValueError(
                    f"Shared LoRA A has shape {tuple(shared_a.shape)}, expected "
                    f"{(rank, base.in_features)}"
                )
            self.lora_A = shared_a
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))

    @property
    def weight(self) -> nn.Parameter:
        """Expose the frozen weight for compatibility with HF internals."""

        return self.base.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = F.linear(F.linear(inputs, self.lora_A), self.lora_B)
        return self.base(inputs) + residual * self.scaling


class Conv2dLoRA(nn.Module):
    """Frozen Conv2d plus rank-r spatial-down/channel-up LoRA residual."""

    def __init__(self, base: nn.Conv2d, *, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if base.groups != 1:
            raise ValueError("Conv2dLoRA currently requires groups=1")
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        kernel_h, kernel_w = base.kernel_size
        self.lora_A = nn.Parameter(
            torch.empty(rank, base.in_channels, kernel_h, kernel_w)
        )
        self.lora_B = nn.Parameter(torch.zeros(base.out_channels, rank, 1, 1))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self) -> nn.Parameter:
        """Expose the frozen weight for compatibility with HF internals."""

        return self.base.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        down = F.conv2d(
            inputs,
            self.lora_A,
            bias=None,
            stride=self.base.stride,
            padding=self.base.padding,
            dilation=self.base.dilation,
            groups=1,
        )
        residual = F.conv2d(down, self.lora_B)
        return self.base(inputs) + residual * self.scaling


@dataclass(frozen=True)
class LoRAInjectionReport:
    """Names of every module replaced by a LoRA wrapper."""

    patch_embedding: str
    query: tuple[str, ...]
    key: tuple[str, ...]
    value: tuple[str, ...]
    output: tuple[str, ...]

    @property
    def all_modules(self) -> tuple[str, ...]:
        return (
            (self.patch_embedding,)
            + self.query
            + self.key
            + self.value
            + self.output
        )


def _freeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def _find_patch_embedding(model: nn.Module) -> tuple[str, nn.Conv2d]:
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
        and (
            name == "embeddings.patch_embeddings"
            or name.endswith(".embeddings.patch_embeddings")
        )
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one DINOv3 Conv2d patch embedding named "
            f"'embeddings.patch_embeddings'; found {[name for name, _ in candidates]}"
        )
    return candidates[0]


def _find_attention_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    blocks = []
    for name, module in model.named_modules():
        if all(
            hasattr(module, child)
            for child in ("q_proj", "k_proj", "v_proj", "o_proj")
        ):
            projections = [
                getattr(module, child)
                for child in ("q_proj", "k_proj", "v_proj", "o_proj")
            ]
            if all(isinstance(projection, nn.Linear) for projection in projections):
                blocks.append((name, module))
    if not blocks:
        raise RuntimeError(
            "No DINOv3 attention blocks with q_proj/k_proj/v_proj/o_proj were found"
        )
    return blocks


def inject_lora_dinov3(
    model: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 16.0,
) -> LoRAInjectionReport:
    """Freeze DINOv3 and inject all LoRA targets required by AnyMC3D."""

    _freeze(model)
    patch_name, patch_module = _find_patch_embedding(model)
    patch_parent_name, patch_attribute = patch_name.rsplit(".", 1)
    patch_parent = model.get_submodule(patch_parent_name)
    setattr(
        patch_parent,
        patch_attribute,
        Conv2dLoRA(patch_module, rank=rank, alpha=alpha),
    )

    target_names: dict[str, list[str]] = {
        "query": [],
        "key": [],
        "value": [],
        "output": [],
    }
    for block_name, attention in _find_attention_blocks(model):
        q_proj = attention.q_proj
        k_proj = attention.k_proj
        v_proj = attention.v_proj
        o_proj = attention.o_proj
        assert isinstance(q_proj, nn.Linear)
        assert isinstance(k_proj, nn.Linear)
        assert isinstance(v_proj, nn.Linear)
        assert isinstance(o_proj, nn.Linear)
        if not (
            q_proj.in_features == k_proj.in_features == v_proj.in_features
        ):
            raise RuntimeError(
                f"{block_name}: Q/K/V input dimensions differ and cannot share "
                "a fused-QKV LoRA A matrix"
            )
        shared_a = nn.Parameter(torch.empty(rank, q_proj.in_features))
        nn.init.kaiming_uniform_(shared_a, a=math.sqrt(5))
        attention.q_proj = LinearLoRA(
            q_proj, rank=rank, alpha=alpha, shared_a=shared_a
        )
        attention.k_proj = LinearLoRA(
            k_proj, rank=rank, alpha=alpha, shared_a=shared_a
        )
        attention.v_proj = LinearLoRA(
            v_proj, rank=rank, alpha=alpha, shared_a=shared_a
        )
        attention.o_proj = LinearLoRA(o_proj, rank=rank, alpha=alpha)
        target_names["query"].append(f"{block_name}.q_proj")
        target_names["key"].append(f"{block_name}.k_proj")
        target_names["value"].append(f"{block_name}.v_proj")
        target_names["output"].append(f"{block_name}.o_proj")

    report = LoRAInjectionReport(
        patch_embedding=patch_name,
        query=tuple(target_names["query"]),
        key=tuple(target_names["key"]),
        value=tuple(target_names["value"]),
        output=tuple(target_names["output"]),
    )
    expected_blocks = int(
        getattr(model.config, "num_hidden_layers", len(report.query))
    )
    if not all(
        len(names) == expected_blocks
        for names in (report.query, report.key, report.value, report.output)
    ):
        raise RuntimeError(
            "Incomplete DINOv3 LoRA injection: "
            f"expected {expected_blocks} of every projection, got "
            f"Q={len(report.query)}, K={len(report.key)}, "
            f"V={len(report.value)}, O={len(report.output)}"
        )
    return report


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return de-duplicated LoRA parameters only."""

    seen: set[int] = set()
    parameters: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, (LinearLoRA, Conv2dLoRA)):
            for parameter in (module.lora_A, module.lora_B):
                if id(parameter) not in seen:
                    parameters.append(parameter)
                    seen.add(id(parameter))
    return parameters


def assert_only_lora_trainable(model: nn.Module) -> None:
    """Fail if a trainable backbone parameter is not a LoRA matrix."""

    allowed = {id(parameter) for parameter in lora_parameters(model)}
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in allowed
    ]
    if unexpected:
        raise RuntimeError(
            f"Non-LoRA backbone parameters are trainable: {unexpected[:20]}"
        )
