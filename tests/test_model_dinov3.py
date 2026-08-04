from __future__ import annotations

import copy

import pytest
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from model_arch.anymc3d_dinov3 import (
    AnyMC3DDINOv3,
    MaskedQueryAttentionPool,
)
from model_arch.lora_dinov3 import Conv2dLoRA, LinearLoRA


def test_transformers_cls_register_patch_order_and_rectangular_input():
    config = DINOv3ViTConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        image_size=64,
        patch_size=16,
        num_register_tokens=4,
    )
    backbone = DINOv3ViTModel(config)
    output = backbone(torch.randn(2, 3, 64, 80))
    # 1 CLS + 4 registers + (4*5) patches.
    assert output.last_hidden_state.shape == (2, 25, 32)
    assert output.pooler_output.shape == (2, 32)


def test_lora_targets_frozen_parameters_and_gradients(tiny_dinov3):
    model = AnyMC3DDINOv3(
        backbone=tiny_dinov3, lora_rank=4, lora_alpha=8
    )
    assert isinstance(model.backbone.embeddings.patch_embeddings, Conv2dLoRA)
    assert len(model.lora_report.query) == 2
    for block in model.backbone.model.layer:
        assert isinstance(block.attention.q_proj, LinearLoRA)
        assert isinstance(block.attention.k_proj, LinearLoRA)
        assert isinstance(block.attention.v_proj, LinearLoRA)
        assert isinstance(block.attention.o_proj, LinearLoRA)
        assert not block.attention.q_proj.base.weight.requires_grad
    volume = torch.rand(2, 4, 1, 64, 80)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    output = model(volume, mask)
    output.logits.sum().backward()
    assert model.backbone.embeddings.patch_embeddings.lora_B.grad is not None
    assert model.backbone.model.layer[0].attention.q_proj.lora_B.grad is not None
    assert model.attention_pool.task_query.grad is not None
    assert model.classification_head.weight.grad is not None
    base_trainable = [
        name
        for name, parameter in model.backbone.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    assert not base_trainable


def test_masked_attention_shape_sum_padding_and_output():
    pool = MaskedQueryAttentionPool(8)
    embeddings = torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    volume, attention = pool(embeddings, mask)
    assert volume.shape == (2, 8)
    assert attention.shape == (2, 4)
    assert torch.all(attention >= 0)
    assert torch.allclose(attention.sum(1), torch.ones(2))
    assert torch.equal(attention[0, 2:], torch.zeros(2))


def test_only_valid_slices_are_encoded_and_chunking_is_equivalent(tiny_dinov3):
    model = AnyMC3DDINOv3(
        backbone=tiny_dinov3,
        lora_rank=2,
        lora_alpha=4,
        slice_chunk_size=2,
    ).eval()
    volume = torch.rand(2, 5, 1, 64, 80)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    with torch.no_grad():
        chunked = model(
            volume, mask, slice_chunk_size=2
        ).logits
        all_at_once = model(
            volume, mask, slice_chunk_size=100
        ).logits
    assert model.last_num_encoded_slices == 8
    assert torch.allclose(chunked, all_at_once, atol=1e-5, rtol=1e-5)


def test_invalid_canvas_fails_without_hidden_resize(tiny_dinov3):
    model = AnyMC3DDINOv3(backbone=tiny_dinov3)
    with pytest.raises(ValueError, match="divisible"):
        model(
            torch.rand(1, 2, 1, 63, 80),
            torch.ones(1, 2, dtype=torch.bool),
        )
