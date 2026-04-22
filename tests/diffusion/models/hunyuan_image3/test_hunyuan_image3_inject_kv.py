# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``ImageKVCacheManager.inject_prompt_kv_cache``.

This is the core interface the KV-reuse PR adds: it takes AR-produced
text KV tensors and pre-populates the DiT's ``image_kv_cache_map`` so
subsequent denoising steps can be run with ``first_step=False``
without recomputing the text KV.

The contract we guard here:

* For a pos-only branch, the cached prefix length equals
  ``pos_len + num_special_tokens`` and the first ``pos_len`` rows of
  the cached tensors equal the input positive tensors verbatim.
* For a pos+neg branch with different lengths, both branches are
  zero-padded to ``max(pos_len, neg_len)`` so they share a single
  ``L_text`` slot, and the returned length equals
  ``max_len + num_special_tokens``.
* The trailing ``num_special_tokens`` + EOI slots are zero-filled --
  this is the layout the attention mask in ``_forward_with_kv_reuse``
  relies on (those positions are masked out of the softmax).
"""

import pytest
import torch

from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    ImageKVCacheManager,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _mgr() -> ImageKVCacheManager:
    # ImageKVCacheManager.__init__ queries the SP world-size, which is
    # only initialised inside a distributed worker. ``inject_prompt_kv_cache``
    # itself does not touch SP state, so bypass __init__ and stamp the
    # attributes the method actually reads.
    mgr = ImageKVCacheManager.__new__(ImageKVCacheManager)
    mgr.num_heads = 4
    mgr.num_kv_heads = 2
    mgr.head_dim = 8
    mgr.scaling = 1.0
    mgr.image_token_len = 16
    mgr.image_kv_cache_map = None
    return mgr


def _rand_kv(length: int, kv_heads: int = 2, head_dim: int = 8):
    k = torch.randn(length, kv_heads, head_dim)
    v = torch.randn(length, kv_heads, head_dim)
    return k, v


def test_inject_pos_only_length_and_layout():
    mgr = _mgr()
    pos_k, pos_v = _rand_kv(length=17)

    cached_len = mgr.inject_prompt_kv_cache(pos_k, pos_v, num_special_tokens=3)

    # Returned prefix length = text + special tokens (excludes eoi).
    assert cached_len == 17 + 3

    cached_k, cached_v = mgr.image_kv_cache_map
    # Full cached layout = pos_text + special + eoi = 17 + 3 + 1 = 21.
    assert cached_k.shape == (21, 2, 8)
    assert cached_v.shape == (21, 2, 8)
    # First pos_len rows preserved verbatim.
    assert torch.equal(cached_k[:17], pos_k)
    assert torch.equal(cached_v[:17], pos_v)
    # The 3 special slots + 1 eoi slot must be zero (they are masked
    # out of the softmax by the attention mask in _forward_with_kv_reuse).
    assert torch.equal(cached_k[17:], torch.zeros(4, 2, 8))
    assert torch.equal(cached_v[17:], torch.zeros(4, 2, 8))


def test_inject_pos_and_neg_same_length():
    mgr = _mgr()
    pos_k, pos_v = _rand_kv(length=10)
    neg_k, neg_v = _rand_kv(length=10)

    cached_len = mgr.inject_prompt_kv_cache(pos_k, pos_v, neg_k, neg_v, num_special_tokens=3)

    assert cached_len == 10 + 3
    cached_k, _ = mgr.image_kv_cache_map
    # 2 * (text + special + eoi) = 2 * (10 + 3 + 1) = 28.
    assert cached_k.shape == (28, 2, 8)
    # Pos branch occupies rows [0:10], neg branch [14:24] (after
    # 10 text + 3 special + 1 eoi).
    assert torch.equal(cached_k[:10], pos_k)
    assert torch.equal(cached_k[14:24], neg_k)


def test_inject_pos_and_neg_mismatched_length_pads_shorter_branch():
    # This is the path that guards against the ``L_pos=6833, L_neg=1``
    # degeneracy: whichever branch is shorter must be zero-padded up to
    # ``L_text = max(L_pos, L_neg)`` and the returned length must
    # reflect the padded max.
    mgr = _mgr()
    pos_k, pos_v = _rand_kv(length=12)
    neg_k, neg_v = _rand_kv(length=7)  # shorter

    cached_len = mgr.inject_prompt_kv_cache(pos_k, pos_v, neg_k, neg_v, num_special_tokens=3)

    assert cached_len == 12 + 3  # max_len + special

    cached_k, cached_v = mgr.image_kv_cache_map
    # Layout: [pos(12) | special(3) | eoi(1) | neg_padded(12) | special(3) | eoi(1)] = 32
    assert cached_k.shape == (32, 2, 8)
    # Positive branch preserved.
    assert torch.equal(cached_k[:12], pos_k)
    # Negative branch: first neg_len rows are the original neg_k, the
    # remaining rows up to L_text=12 must be zero padding.
    assert torch.equal(cached_k[16:16 + 7], neg_k)
    assert torch.equal(cached_k[16 + 7:16 + 12], torch.zeros(5, 2, 8))
    # Values follow the same layout.
    assert torch.equal(cached_v[:12], pos_v)
    assert torch.equal(cached_v[16:16 + 7], neg_v)


def test_inject_neg_longer_than_pos_also_pads():
    # Symmetric case: explicit negative prompt longer than the positive
    # one. Must pad pos, not neg.
    mgr = _mgr()
    pos_k, pos_v = _rand_kv(length=5)
    neg_k, neg_v = _rand_kv(length=9)

    cached_len = mgr.inject_prompt_kv_cache(pos_k, pos_v, neg_k, neg_v, num_special_tokens=3)

    assert cached_len == 9 + 3
    cached_k, _ = mgr.image_kv_cache_map
    # 2 * (9 + 3 + 1) = 26
    assert cached_k.shape == (26, 2, 8)
    assert torch.equal(cached_k[:5], pos_k)
    assert torch.equal(cached_k[5:9], torch.zeros(4, 2, 8))  # pos padding
    assert torch.equal(cached_k[13:13 + 9], neg_k)


def test_inject_custom_num_special_tokens():
    mgr = _mgr()
    pos_k, pos_v = _rand_kv(length=4)

    cached_len = mgr.inject_prompt_kv_cache(pos_k, pos_v, num_special_tokens=5)

    assert cached_len == 4 + 5
    cached_k, _ = mgr.image_kv_cache_map
    assert cached_k.shape == (4 + 5 + 1, 2, 8)


def test_inject_preserves_dtype_and_device():
    mgr = _mgr()
    pos_k = torch.randn(6, 2, 8, dtype=torch.float16)
    pos_v = torch.randn(6, 2, 8, dtype=torch.float16)

    mgr.inject_prompt_kv_cache(pos_k, pos_v, num_special_tokens=3)

    cached_k, cached_v = mgr.image_kv_cache_map
    assert cached_k.dtype == torch.float16
    assert cached_v.dtype == torch.float16
    assert cached_k.device == pos_k.device
