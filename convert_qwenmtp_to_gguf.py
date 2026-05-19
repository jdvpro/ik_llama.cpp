#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert Qwen3-Next / Qwen3.5-MoE (incl. Qwen3.6 VL) HF models to GGUF, with MTP.

Targets the tensor layout that the ik_llama.cpp runtime in this tree already
supports, matching the working reference GGUF (e.g. Qwen3.6-35B-A3B-iq4_xs.gguf):

  * arch = qwen3next   for Qwen3NextForCausalLM            (Qwen3-Next 80B, ...)
  * arch = qwen35moe   for Qwen3_5MoeForConditionalGeneration / Qwen3_5MoeForCausalLM
                       (Qwen3.5/3.6-MoE 35B; vision tensors are skipped)

MTP weights are written as an additional "block" of n_main_layers + n_mtp_idx,
following the existing GGUF schema:
    blk.{N}.nextn.eh_proj.weight              # mtp.fc.weight
    blk.{N}.nextn.enorm.weight                # mtp.pre_fc_norm_embedding.weight (+1)
    blk.{N}.nextn.hnorm.weight                # mtp.pre_fc_norm_hidden.weight    (+1)
    blk.{N}.nextn.shared_head_norm.weight     # mtp.norm.weight                  (+1)
    blk.{N}.{attn_*,ffn_*,...}                # full transformer block of mtp.layers.0

Usage:
    python convert_qwenmtp_to_gguf.py /path/to/hf/model --outtype bf16 \
        --outfile /path/to/output.gguf

This script is a thin wrapper over convert_hf_to_gguf.py: it forces use of the
system-installed gguf-py (which already knows about qwen3next/qwen35moe), then
registers two model classes for the Qwen3-Next / Qwen3.5-MoE architectures.
"""
from __future__ import annotations

import os
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

# Make convert_hf_to_gguf importable from the same directory.
# convert_hf_to_gguf itself inserts gguf-py/ into sys.path before importing gguf,
# so the local gguf-py (with our QWEN3NEXT / QWEN35MOE additions) is used.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import convert_hf_to_gguf as _base
import gguf  # local gguf-py (already inserted by convert_hf_to_gguf import)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _scan_tensor_names(dir_model: Path) -> set[str]:
    """Return the set of all tensor names in a HF safetensors checkpoint without
    loading any tensor data. Reads model.safetensors.index.json for sharded
    checkpoints, or opens the single shard's header for unsharded ones."""
    idx_path = dir_model / "model.safetensors.index.json"
    if idx_path.exists():
        with open(idx_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        wm = idx.get("weight_map") or {}
        return set(wm.keys())
    # Single-file fallback
    from safetensors import safe_open
    single = dir_model / "model.safetensors"
    if not single.exists():
        # No safetensors at all (e.g. pytorch_model.bin) -> caller should
        # fall back to slow scan. For our target models this never happens.
        return set()
    with safe_open(str(single), framework="pt") as st:
        return set(st.keys())


def _count_mtp_layers_from_index(dir_model: Path) -> int:
    """Count distinct N values found in tensor names matching mtp.layers.N.*"""
    layers: set[int] = set()
    pat = re.compile(r"^mtp\.layers\.(\d+)\.")
    for k in _scan_tensor_names(dir_model):
        m = pat.match(k)
        if m:
            layers.add(int(m.group(1)))
    return len(layers)


def _has_language_model_prefix(dir_model: Path) -> bool:
    """VL-style Qwen3.5-MoE checkpoints (e.g. Qwen3.6-35B-A3B) wrap the LM under
    model.language_model.* . Pure CausalLM checkpoints (e.g. Qwen3-Next 80B) use
    model.* directly. Detected by inspecting any tensor name."""
    for k in _scan_tensor_names(dir_model):
        if k.startswith("model.language_model."):
            return True
        if k.startswith("model.layers."):
            return False
    return False


# ---------------------------------------------------------------------------
# Qwen3NextForCausalLM  ->  arch qwen3next
# ---------------------------------------------------------------------------
@_base.Model.register("Qwen3NextForCausalLM")
class Qwen3NextModel(_base.Qwen2MoeModel):
    """Qwen3-Next dense+linear hybrid MoE with optional MTP head.

    Linear-attention layers use fused in_proj_qkvz (split here into ATTN_QKV +
    ATTN_GATE) and fused in_proj_ba (kept as SSM_BETA_ALPHA / blk.N.ssm_ba).
    """
    model_arch = gguf.MODEL_ARCH.QWEN3NEXT

    # Will be set in __init__
    n_main_layers: int = 0
    n_mtp_layers: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Extend block_count to cover MTP layer(s) so tensor_map and the GGUF
        # writer reserve metadata/space for them.
        self.n_main_layers = int(self.hparams["num_hidden_layers"])
        self.n_mtp_layers = _count_mtp_layers_from_index(self.dir_model)
        if self.n_mtp_layers > 0:
            self.block_count = self.n_main_layers + self.n_mtp_layers
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    # ------------------------------------------------------------------
    # GGUF metadata
    # ------------------------------------------------------------------
    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        h = self.hparams
        # SSM / linear-attention metadata (mirrors fork's mapping; see also
        # llama.cpp upstream upstream PR thread).
        if "linear_conv_kernel_dim" in h:
            self.gguf_writer.add_ssm_conv_kernel(h["linear_conv_kernel_dim"])
        if "linear_key_head_dim" in h:
            self.gguf_writer.add_ssm_state_size(h["linear_key_head_dim"])
        if "linear_num_key_heads" in h:
            self.gguf_writer.add_ssm_group_count(h["linear_num_key_heads"])
        if "linear_num_value_heads" in h:
            self.gguf_writer.add_ssm_time_step_rank(h["linear_num_value_heads"])
        if "linear_value_head_dim" in h and "linear_num_value_heads" in h:
            self.gguf_writer.add_ssm_inner_size(
                h["linear_value_head_dim"] * h["linear_num_value_heads"]
            )
        # full_attention_interval: writer method may not exist on older gguf-py
        if hasattr(self.gguf_writer, "add_full_attention_interval"):
            self.gguf_writer.add_full_attention_interval(h.get("full_attention_interval", 4))
        else:
            # Fall back to raw uint32 KV
            arch_name = gguf.MODEL_ARCH_NAMES[self.model_arch]
            self.gguf_writer.add_uint32(
                f"{arch_name}.full_attention_interval",
                int(h.get("full_attention_interval", 4)),
            )

        # MTP / NextN layer count
        if self.n_mtp_layers > 0:
            self.gguf_writer.add_nextn_predict_layers(self.n_mtp_layers)

        # Partial RoPE (Qwen3-Next / Qwen3.5)
        if (rope_dim := h.get("head_dim")) is None:
            rope_dim = h["hidden_size"] // h["num_attention_heads"]
        prf = h.get("partial_rotary_factor", 0.25)
        self.gguf_writer.add_rope_dimension_count(int(rope_dim * prf))

        # RoPE base frequency. Qwen3.6 VL stores it under
        # text_config.rope_parameters.rope_theta; older variants keep it at
        # the (flattened) top level. The base Qwen2MoeModel writer only looks
        # at top-level rope_theta, so we explicitly emit it here when found.
        rope_theta = None
        rp = h.get("rope_parameters")
        if isinstance(rp, dict):
            rope_theta = rp.get("rope_theta")
        if rope_theta is None:
            rope_theta = h.get("rope_theta")
        if rope_theta is not None:
            self.gguf_writer.add_rope_freq_base(float(rope_theta))

        # M-RoPE sections (VL variants only). HF stores them as 3 ints
        # [t, h, w] under either rope_parameters.mrope_section (Qwen3.6 VL)
        # or rope_scaling.mrope_section (older naming); ik_llama runtime
        # expects 4 ints (pad with 0). Pure-text checkpoints have no
        # mrope_section -> skip silently.
        mrope_section = None
        for container_key in ("rope_parameters", "rope_scaling"):
            container = h.get(container_key) or {}
            if isinstance(container, dict) and container.get("mrope_section"):
                mrope_section = container["mrope_section"]
                break
        if mrope_section:
            sections = list(mrope_section) + [0] * (4 - len(mrope_section))
            self.gguf_writer.add_rope_dimension_sections(sections[:4])

        # MoE expert intermediate sizes (Qwen3-Next/3.5 use "moe_intermediate_size",
        # base Qwen2MoeModel already emits this if present but be defensive).
        if (msz := h.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(msz)
        if (ssz := h.get("shared_expert_intermediate_size")) is not None:
            self.gguf_writer.add_expert_shared_feed_forward_length(ssz)

    # ------------------------------------------------------------------
    # Tensor transforms
    # ------------------------------------------------------------------
    def modify_tensors(self, data_torch: torch.Tensor, name: str, bid: int | None) -> Iterable[tuple[str, torch.Tensor]]:
        # MTP block: separate code path so we can re-route through modify_tensors
        # with rewritten names at the synthetic block id.
        if name.startswith("mtp."):
            yield from self._modify_mtp_tensors(data_torch, name)
            return

        # Linear-attention specific transforms
        if name.endswith(".A_log"):
            # A_log -> -exp(A) for SSM_A (matches upstream / fork)
            data_torch = -torch.exp(data_torch.float())
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.SSM_A, bid, suffix=""), data_torch)
            return

        if name.endswith(".dt_bias"):
            # HF stores dt_bias at .dt_bias; ik_llama runtime expects ssm_dt.bias.
            # The tensor_map only knows linear_attn.dt_proj.bias, so rename and
            # fall through so the standard map applies.
            new_name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"
            # super().modify_tensors will route to SSM_DT.bias
            yield from super().modify_tensors(data_torch, new_name, bid)
            return

        if name.endswith("linear_attn.conv1d.weight"):
            # squeeze (n_inner, 1, ksize) -> (n_inner, ksize)
            data_torch = data_torch.squeeze()
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.SSM_CONV1D, bid, suffix=".weight"), data_torch)
            return

        # in_proj_qkvz (fused QKV+Z, Qwen3-Next): split into ATTN_QKV + ATTN_GATE.
        # NB: the system gguf-py tensor_map sends this to SSM_IN which is NOT what
        # the ik_llama runtime expects (it reads attn_qkv + attn_gate).
        if "linear_attn.in_proj_qkvz.weight" in name:
            yield from self._split_qkvz(data_torch, bid)
            return

        # in_proj_qkv (separate, Qwen3.5/3.6) -> ATTN_QKV
        if "linear_attn.in_proj_qkv.weight" in name:
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_QKV, bid, suffix=".weight"), data_torch)
            return

        # RMSNorm +1 offset for all other norm.weight tensors EXCEPT
        # linear_attn.norm.weight (matches reference output).
        if name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
            data_torch = data_torch + 1

        # All remaining tensors: defer to Qwen2MoeModel (handles per-expert
        # stacking and standard name mapping via tensor_map).
        yield from super().modify_tensors(data_torch, name, bid)

    # ------------------------------------------------------------------
    # Per-tensor quant overrides
    # ------------------------------------------------------------------
    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # The ik_llama SSM_CONV operator only accepts F32 weights
        # (asserts src2->nb[0] == sizeof(float) in ggml_compute_forward_ssm_conv_f32).
        # ssm_conv1d.weight is 2D after squeeze, so the generic n_dims<=1
        # fast-path that auto-promotes 1D tensors to F32 does NOT cover it,
        # and at --outtype bf16 it would otherwise be stored as BF16.
        if bid is not None and new_name == self.format_tensor_name(
                gguf.MODEL_TENSOR.SSM_CONV1D, bid, suffix=".weight"):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    # ------------------------------------------------------------------
    # in_proj_qkvz split + V-head reorder
    # ------------------------------------------------------------------
    def _split_qkvz(self, data: torch.Tensor, bid: int | None) -> Iterable[tuple[str, torch.Tensor]]:
        """Split fused in_proj_qkvz weight into ATTN_QKV (q+k+v) and ATTN_GATE (z),
        and reorder from grouped-by-Khead to interleaved-by-Khead layout so ggml
        broadcast works. See fork's Qwen3NextModel.modify_tensors for derivation."""
        h = self.hparams
        head_k_dim = h["linear_key_head_dim"]
        head_v_dim = h["linear_value_head_dim"]
        num_v_heads = h["linear_num_value_heads"]
        num_k_heads = h["linear_num_key_heads"]
        hidden_size = h["hidden_size"]
        if num_v_heads % num_k_heads != 0:
            raise ValueError(
                f"linear_num_value_heads ({num_v_heads}) must be divisible by "
                f"linear_num_key_heads ({num_k_heads})"
            )
        v_per_k = num_v_heads // num_k_heads
        split_sizes = [
            head_k_dim,           # q partition (per K-head)
            head_k_dim,           # k partition (per K-head)
            v_per_k * head_v_dim, # v partition (per K-head; v_per_k V-heads grouped)
            v_per_k * head_v_dim, # z partition (per K-head; v_per_k V-heads grouped)
        ]
        total = sum(split_sizes)
        # HF stores weight as (out_features, in_features) = (sum_per_k * num_k_heads, hidden_size)
        # Permute to (hidden_size, out_features) for view manipulation
        x = data.permute(1, 0).contiguous()
        # View as (hidden_size, num_k_heads, total_per_k)
        x = x.view(-1, num_k_heads, total)
        q, k, v, z = torch.split(x, split_sizes, dim=-1)
        # Flatten per-K-head dimension back into per-tensor out dim
        q = q.contiguous().view(hidden_size, -1)
        k = k.contiguous().view(hidden_size, -1)
        v = v.contiguous().view(hidden_size, -1)
        z = z.contiguous().view(hidden_size, -1)
        # Recombine q,k,v -> (hidden_size, out_qkv), permute back to (out, hidden)
        qkv = torch.cat([q, k, v], dim=-1).permute(1, 0).contiguous()
        z = z.permute(1, 0).contiguous()
        yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_QKV,  bid, suffix=".weight"), qkv)
        yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_GATE, bid, suffix=".weight"), z)

    # ------------------------------------------------------------------
    # MTP tensor remap
    # ------------------------------------------------------------------
    def _modify_mtp_tensors(self, data_torch: torch.Tensor, name: str) -> Iterable[tuple[str, torch.Tensor]]:
        """Remap HF mtp.* tensors onto the synthetic block at bid = n_main + mtp_idx."""
        n_main = self.n_main_layers
        m = re.match(r"^mtp\.layers\.(\d+)\.", name)
        mtp_idx = int(m.group(1)) if m else 0
        bid = n_main + mtp_idx

        # NEXTN-head specific tensors (all live at the start of the MTP block)
        if name == "mtp.fc.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_EH_PROJ, bid, suffix=".weight"), data_torch)
            return
        if name == "mtp.pre_fc_norm_embedding.weight":
            data_torch = data_torch + 1
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_ENORM, bid, suffix=".weight"), data_torch)
            return
        if name == "mtp.pre_fc_norm_hidden.weight":
            data_torch = data_torch + 1
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_HNORM, bid, suffix=".weight"), data_torch)
            return
        if name == "mtp.norm.weight":
            data_torch = data_torch + 1
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_SHARED_HEAD_NORM, bid, suffix=".weight"), data_torch)
            return

        # All other mtp.layers.M.X tensors: rewrite to model.layers.{bid}.X and
        # re-enter the regular modify_tensors pipeline with the correct bid.
        # +1 RMSNorm offset is applied by modify_tensors for non-linear_attn norms.
        layer_name = re.sub(r"^mtp\.layers\.\d+\.", f"model.layers.{bid}.", name)
        yield from self.modify_tensors(data_torch, layer_name, bid)


# ---------------------------------------------------------------------------
# Qwen3.5-MoE common base: V-head reorder for asymmetric num_k < num_v
# ---------------------------------------------------------------------------
class _LinearAttentionVReorderBase(Qwen3NextModel):
    """Reorders V heads from grouped-by-K to tiled order so ggml_repeat can be
    used instead of an interleaved repeat. Only meaningful when
    linear_num_value_heads != linear_num_key_heads (Qwen3.5 has 16 K, 32 V).

    See fork class of the same name for the reference and the upstream
    discussion at https://github.com/ggml-org/llama.cpp/pull/19468 .
    """
    model_arch = gguf.MODEL_ARCH.QWEN3NEXT  # overridden by subclass

    @staticmethod
    def _reorder_v_heads(tensor: torch.Tensor, dim: int, num_k_heads: int,
                         num_v_per_k: int, head_dim: int) -> torch.Tensor:
        shape = list(tensor.shape)
        if dim < 0:
            dim += len(shape)
        new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1:]
        tensor = tensor.reshape(*new_shape)
        perm = list(range(len(new_shape)))
        perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
        return tensor.permute(*perm).contiguous().reshape(*shape)

    def modify_tensors(self, data_torch, name, bid):
        num_k_heads = self.hparams.get("linear_num_key_heads", 0) or 0
        num_v_heads = self.hparams.get("linear_num_value_heads", 0) or 0
        if (num_k_heads and num_v_heads and num_k_heads != num_v_heads
                and "linear_attn." in name):
            head_k_dim = self.hparams["linear_key_head_dim"]
            head_v_dim = self.hparams["linear_value_head_dim"]
            v_per_k = num_v_heads // num_k_heads

            if ".in_proj_qkv." in name:
                # Reorder only the V rows of the [q | k | v] stack
                q_dim = head_k_dim * num_k_heads
                k_dim = head_k_dim * num_k_heads
                q = data_torch[:q_dim]
                k = data_torch[q_dim:q_dim + k_dim]
                v = data_torch[q_dim + k_dim:]
                v = self._reorder_v_heads(v, 0, num_k_heads, v_per_k, head_v_dim)
                data_torch = torch.cat([q, k, v], dim=0)
            elif ".in_proj_z." in name:
                data_torch = self._reorder_v_heads(data_torch, 0, num_k_heads, v_per_k, head_v_dim)
            elif ".in_proj_a." in name or ".in_proj_b." in name:
                # Per-head scalar parameters (head_dim = 1)
                data_torch = self._reorder_v_heads(data_torch, 0, num_k_heads, v_per_k, 1)
            elif ".A_log" in name or ".dt_bias" in name or ".dt_proj" in name:
                if data_torch.ndim == 1:
                    data_torch = self._reorder_v_heads(
                        data_torch.unsqueeze(-1), 0, num_k_heads, v_per_k, 1
                    ).squeeze(-1)
                else:
                    data_torch = self._reorder_v_heads(data_torch, -1, num_k_heads, v_per_k, 1)
            elif ".conv1d" in name:
                # qk channels first, v channels last; reorder only v portion
                data = data_torch.squeeze()
                qk_channels = head_k_dim * num_k_heads * 2
                qk_part = data[:qk_channels]
                v_part = data[qk_channels:]
                v_part = self._reorder_v_heads(v_part, 0, num_k_heads, v_per_k, head_v_dim)
                data_torch = torch.cat([qk_part, v_part], dim=0)
                # Re-add the squeezed dim is unnecessary; downstream squeeze handler
                # is in Qwen3NextModel.modify_tensors which does squeeze() again on
                # 3-D conv kernels. Our data here is already 2-D, but the parent
                # call site will pass through fine since squeeze() is a no-op.
            elif ".out_proj." in name:
                data_torch = self._reorder_v_heads(data_torch, 1, num_k_heads, v_per_k, head_v_dim)

        yield from super().modify_tensors(data_torch, name, bid)


# ---------------------------------------------------------------------------
# Qwen3_5MoeForConditionalGeneration / Qwen3_5MoeForCausalLM  ->  arch qwen35moe
# ---------------------------------------------------------------------------
@_base.Model.register("Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM")
class Qwen3_5MoeModel(_LinearAttentionVReorderBase):
    """Qwen3.5-MoE 35B (incl. Qwen3.6-35B-A3B VL variant) with optional MTP.

    For the VL variant (Qwen3_5MoeForConditionalGeneration):
      * the config is nested: text params live under config['text_config']
      * tensors are prefixed model.language_model.* ; we strip that
      * model.visual.* tensors are skipped (text-only GGUF, matching the
        existing reference layout)
      * MoE expert weights are packed (mlp.experts.gate_up_proj +
        mlp.experts.down_proj); we split/permute them here
    """
    model_arch = gguf.MODEL_ARCH.QWEN35MOE
    _vl_prefix: bool = False

    def __init__(self, *args, **kwargs):
        # Pre-flatten nested text_config into hparams BEFORE the base __init__
        # reads num_hidden_layers etc. We can't easily intercept hparams loading
        # from here, so we patch the loaded dict in-place after super().__init__()
        # and recompute block_count / tensor_map. The base __init__ tolerates
        # 'num_hidden_layers' present at either level because find_hparam looks
        # in self.hparams; nested configs cause the find to fail, so we run a
        # quick pre-check and load+flatten manually if needed.
        dir_model = kwargs.get("dir_model") or (args[0] if args else None)
        if dir_model is not None and isinstance(dir_model, Path):
            cfg_path = dir_model / "config.json"
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if "text_config" in raw and "num_hidden_layers" not in raw:
                    # Inject text_config keys into the top level temporarily so
                    # the base loader picks them up. We do this by writing back
                    # a flat shadow copy held only in memory - but Model.load_hparams
                    # reads the file directly. To work around that, we monkeypatch
                    # Model.load_hparams for this instance's first call.
                    self._needs_flatten = True
                    self._raw_cfg = raw
                else:
                    self._needs_flatten = False
                self._is_vl = "vision_config" in raw or "visual" in str(raw.get("architectures", []))
            else:
                self._needs_flatten = False
                self._is_vl = False
        # Patch Model.load_hparams temporarily if needed so the base init works
        if getattr(self, "_needs_flatten", False):
            orig_load = _base.Model.load_hparams
            def _patched_load(d):
                hp = orig_load(d)
                if "text_config" in hp:
                    flat = dict(hp)
                    flat.update(hp["text_config"])
                    return flat
                return hp
            _base.Model.load_hparams = staticmethod(_patched_load)
            try:
                super().__init__(*args, **kwargs)
            finally:
                _base.Model.load_hparams = staticmethod(orig_load)
        else:
            super().__init__(*args, **kwargs)

        # Detect whether the checkpoint uses model.language_model.* prefix
        self._vl_prefix = _has_language_model_prefix(self.dir_model)

    # ------------------------------------------------------------------
    def set_gguf_parameters(self):
        # Some VL configs put rope_theta/etc. only in text_config; we've already
        # flattened. Just call the parent.
        super().set_gguf_parameters()

    # ------------------------------------------------------------------
    # Tensor transforms
    # ------------------------------------------------------------------
    def modify_tensors(self, data_torch, name, bid):
        # Skip vision tensors entirely (this script produces a text-only LM GGUF;
        # vision projector should be exported separately via convert script's
        # mmproj path if needed).
        if name.startswith("model.visual.") or name.startswith("visual."):
            return

        # Strip the VL language_model. prefix so the rest of the pipeline (and
        # the system tensor_map) sees standard model.layers.N.* names.
        if self._vl_prefix:
            name = name.replace("language_model.", "")

        # MTP block handling (must come before generic decoder remap)
        if name.startswith("mtp."):
            yield from self._modify_mtp_tensors_qwen35(data_torch, name)
            return

        # Packed expert tensors (Qwen3.6-VL style)
        if name.endswith("mlp.experts.gate_up_proj") or name.endswith("mlp.experts.gate_up_proj.weight"):
            yield from self._split_packed_gate_up(data_torch, name, bid)
            return
        if name.endswith("mlp.experts.down_proj") or name.endswith("mlp.experts.down_proj.weight"):
            yield from self._handle_packed_down(data_torch, name, bid)
            return

        # All other tensors -> _LinearAttentionVReorderBase.modify_tensors ->
        # Qwen3NextModel.modify_tensors -> Qwen2MoeModel.modify_tensors
        yield from super().modify_tensors(data_torch, name, bid)

    # ------------------------------------------------------------------
    def _split_packed_gate_up(self, data_torch, name, bid):
        """Split packed gate_up_proj into gate_proj + up_proj.

        Verified against HF Qwen3.6-35B-A3B (VL):
            gate_up_proj: (n_expert, 2*n_ff_exp, n_embd)
        Target GGML ne:  {n_embd, n_ff_exp, n_expert}
        Target PyTorch:  (n_expert, n_ff_exp, n_embd)
        So we split along axis 1 (the 2*n_ff_exp dim); no permute needed.
        """
        if data_torch.ndim < 3 or data_torch.shape[1] % 2 != 0:
            raise ValueError(f"Unexpected gate_up_proj shape for {name}: {tuple(data_torch.shape)}")
        split_dim = data_torch.shape[1] // 2
        gate = data_torch[:, :split_dim, :].contiguous()
        up   = data_torch[:, split_dim:, :].contiguous()
        # Synthesize the canonical "stacked-experts" name expected by the tensor_map.
        base_name = name.removesuffix(".weight").rsplit(".", 1)[0]
        gate_name = f"{base_name}.gate_proj.weight"
        up_name = f"{base_name}.up_proj.weight"
        yield (self.map_tensor_name(gate_name), gate)
        yield (self.map_tensor_name(up_name), up)

    def _handle_packed_down(self, data_torch, name, bid):
        """Pass-through for packed down_proj.

        Verified against HF Qwen3.6-35B-A3B (VL):
            down_proj: (n_expert, n_embd, n_ff_exp)
        Target GGML ne:  {n_ff_exp, n_embd, n_expert}
        Target PyTorch:  (n_expert, n_embd, n_ff_exp)
        Already in the right layout; no permute.
        """
        mapped = f"{name}.weight" if not name.endswith(".weight") else name
        yield (self.map_tensor_name(mapped), data_torch.contiguous())

    # ------------------------------------------------------------------
    def _modify_mtp_tensors_qwen35(self, data_torch, name):
        """Same routing as Qwen3NextModel._modify_mtp_tensors, but ensures that
        once we re-enter modify_tensors with the rewritten name, we hit THIS
        class's overrides (e.g. packed expert handling for VL)."""
        n_main = self.n_main_layers
        m = re.match(r"^mtp\.layers\.(\d+)\.", name)
        mtp_idx = int(m.group(1)) if m else 0
        bid = n_main + mtp_idx

        if name == "mtp.fc.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_EH_PROJ, bid, suffix=".weight"), data_torch)
            return
        if name == "mtp.pre_fc_norm_embedding.weight":
            data_torch = data_torch + 1
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_ENORM, bid, suffix=".weight"), data_torch)
            return
        if name == "mtp.pre_fc_norm_hidden.weight":
            data_torch = data_torch + 1
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_HNORM, bid, suffix=".weight"), data_torch)
            return
        if name == "mtp.norm.weight":
            data_torch = data_torch + 1
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_SHARED_HEAD_NORM, bid, suffix=".weight"), data_torch)
            return

        # Everything else: remap mtp.layers.N.X -> model.layers.{bid}.X and
        # re-enter our own modify_tensors so packed-experts handling kicks in.
        layer_name = re.sub(r"^mtp\.layers\.\d+\.", f"model.layers.{bid}.", name)
        yield from self.modify_tensors(data_torch, layer_name, bid)


# ---------------------------------------------------------------------------
# Entry point: delegate to the base CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _base.main()
