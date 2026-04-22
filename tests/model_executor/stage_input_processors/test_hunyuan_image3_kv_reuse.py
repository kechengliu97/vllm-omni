# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the HunyuanImage3 AR->DiT KV-reuse stage input processor.

Covers the two new public entry points added by the KV-reuse PR:

* ``expand_cfg_prompts`` -- builds the CFG companion request.  Guards the
  core invariants that prevent the previously-observed ``L_pos=6833,
  L_neg=1`` degeneracy: (a) the companion must mirror the positive
  prompt's structure, (b) it must inherit the multi-modal data and
  image/system-prompt toggles, (c) ``max_tokens`` must be forced to 1 so
  the companion stops right after prefill, and (d) non-image prompts
  must NOT trigger an expansion.

* ``collect_cfg_kv_caches`` -- the DiT-side collector that pulls
  companion KV back via the ``OmniKVTransferManager``.  We test the
  success path, the "transfer manager returned nothing" path, and the
  exception-swallowing path (collector must never break the parent
  request).
"""

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.stage_input_processors.hunyuan_image3 import (
    CFG_TEXT_SUFFIX,
    collect_cfg_kv_caches,
    expand_cfg_prompts,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# -----------------------------------------------------------------------------
# expand_cfg_prompts
# -----------------------------------------------------------------------------


def _image_prompt(**overrides):
    base = {
        "prompt": "<|startoftext|>SYS\n\nUser: [<img>]paint a cat\n\nAssistant: <think>",
        "user_prompt": "paint a cat",
        "modalities": ["image"],
        "multi_modal_data": {"image": object()},
        "height": 1024,
        "width": 1024,
        "use_system_prompt": True,
    }
    base.update(overrides)
    return base


def test_expand_cfg_prompts_non_dict_returns_empty():
    assert expand_cfg_prompts("just a string", SimpleNamespace()) == []


def test_expand_cfg_prompts_non_image_returns_empty():
    # Text-only prompts must not trigger CFG expansion (this is the
    # short-circuit that keeps t2t / i2t paths cheap).
    txt = {"prompt": "hi", "modalities": []}
    assert expand_cfg_prompts(txt, SimpleNamespace()) == []


def test_expand_cfg_prompts_substitutes_user_text_with_cfg_token():
    prompt = _image_prompt()

    result = expand_cfg_prompts(prompt, SimpleNamespace())

    assert len(result) == 1
    companion = result[0]
    # Role + suffix are the contract the DiT collector relies on.
    assert companion.role == "cfg_text"
    assert companion.request_id_suffix == CFG_TEXT_SUFFIX
    # max_tokens must be forced to 1 -- otherwise the companion would
    # decode an entire CoT for the uncond branch.
    assert companion.sampling_params_override == {"max_tokens": 1}

    neg_dict = companion.prompt
    # <cfg> must replace ONLY the user-text substring; every other
    # section (system, image marker, assistant/trigger) is preserved.
    assert "<cfg>" in neg_dict["prompt"]
    assert "paint a cat" not in neg_dict["prompt"]
    assert neg_dict["prompt"].startswith("<|startoftext|>SYS")
    assert "[<img>]" in neg_dict["prompt"]
    assert "Assistant: <think>" in neg_dict["prompt"]


def test_expand_cfg_prompts_inherits_multimodal_and_meta():
    mm = {"image": object()}
    prompt = _image_prompt(multi_modal_data=mm, height=512, width=768, use_system_prompt=False)

    result = expand_cfg_prompts(prompt, SimpleNamespace())
    neg = result[0].prompt

    assert neg["modalities"] == ["image"]
    # Must carry the same image payload so the uncond branch prefills the image section too.
    assert neg["multi_modal_data"] is mm
    assert neg["height"] == 512
    assert neg["width"] == 768
    assert neg["use_system_prompt"] is False


def test_expand_cfg_prompts_missing_user_text_falls_back_to_positive():
    # When user_prompt is absent we cannot locate the user-text substring,
    # so the fallback keeps the positive prompt verbatim (still structurally
    # valid, <cfg> is not injected). This branch must not raise.
    prompt = {
        "prompt": "<|startoftext|>hi",
        "modalities": ["image"],
    }
    result = expand_cfg_prompts(prompt, SimpleNamespace())
    assert len(result) == 1
    assert result[0].prompt["prompt"] == "<|startoftext|>hi"


def test_expand_cfg_prompts_respects_explicit_negative_prompt():
    # If the caller supplied an explicit negative prompt, it wins over
    # the <cfg>-substitution path.
    prompt = _image_prompt()
    sp = SimpleNamespace(extra_args={"negative_prompt": "blurry, low quality"})

    result = expand_cfg_prompts(prompt, sp)
    assert len(result) == 1
    neg_text = result[0].prompt["prompt"]
    assert "blurry, low quality" in neg_text


# -----------------------------------------------------------------------------
# collect_cfg_kv_caches
# -----------------------------------------------------------------------------


class _FakeKVTransferManager:
    """Minimal stand-in for OmniKVTransferManager."""

    def __init__(self, payload_by_rid):
        self._payload = payload_by_rid
        self.calls = []

    def receive_kv_cache_for_request(self, rid, target_device):
        self.calls.append((rid, target_device))
        return self._payload.get(rid, (None, 0))


def test_collect_cfg_kv_caches_returns_past_kv_and_metadata():
    rid = "req_0"
    companion = "req_0__cfg_text"
    layer_blocks = {"layer_0": object(), "layer_1": object()}
    metadata = {"shape": [1, 2, 3]}
    mgr = _FakeKVTransferManager(
        {companion: ({"layer_blocks": layer_blocks, "metadata": metadata}, 999)}
    )

    out = collect_cfg_kv_caches(rid, {"cfg_text": companion}, mgr, target_device=None)

    assert "cfg_text_past_key_values" in out
    kv = out["cfg_text_past_key_values"]
    # Must be a namespace-like object exposing the layer attributes by name.
    assert getattr(kv, "layer_0") is layer_blocks["layer_0"]
    assert getattr(kv, "layer_1") is layer_blocks["layer_1"]
    assert out["cfg_text_kv_metadata"] == metadata
    assert mgr.calls == [(companion, None)]


def test_collect_cfg_kv_caches_empty_payload_is_silent():
    # When the transfer manager has nothing, the collector must not raise
    # and must return an empty dict so the DiT can fall back to the
    # no-CFG path.
    mgr = _FakeKVTransferManager({})
    out = collect_cfg_kv_caches("req_0", {"cfg_text": "req_0__cfg_text"}, mgr)
    assert out == {}


def test_collect_cfg_kv_caches_swallows_transfer_errors():
    class _BrokenMgr:
        def receive_kv_cache_for_request(self, rid, target_device):
            raise RuntimeError("transfer went sideways")

    # Must not propagate -- the parent DiT request should still run.
    out = collect_cfg_kv_caches("req_0", {"cfg_text": "req_0__cfg_text"}, _BrokenMgr())
    assert out == {}
