# Copyright (c) 2025.
# VLA 使用 HuggingFace PaliGemma (VLM) + Gemma expert (action head) 实现。
# 参考 openpi/Pi0：PaliGemma 处理 image+text (prefix)，Gemma expert 处理 state+action+timestep (suffix)。
# Flow matching 训练：预测速度场 u_t = noise - actions。
#
# 关键设计：action expert 的 num_layers / num_kv_heads / head_dim 与 VLM language model 对齐，
# 这样 VLM forward 产出的 past_key_values 可以直接作为 expert 的 prefix KV cache，无需额外投影。

from __future__ import annotations

import logging
import math
from functools import partial
from typing import Any, List, Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from a1.config import ModelConfig
from a1.vla.affordvla import ProprioProjector

log = logging.getLogger(__name__)

try:
    from transformers import PaliGemmaForConditionalGeneration
except ImportError:
    PaliGemmaForConditionalGeneration = None

try:
    from transformers import GemmaForCausalLM, GemmaConfig
except ImportError:
    GemmaForCausalLM = None
    GemmaConfig = None

try:
    from transformers import AutoProcessor
except ImportError:
    AutoProcessor = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float = 4e-3, max_period: float = 4.0,
) -> torch.Tensor:
    """Pi0-style sine-cosine positional embedding for scalar timesteps."""
    device = time.device
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float64, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].to(torch.float64)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1).to(time.dtype)


def make_att_2d_masks(pad_masks, att_masks):
    """From big_vision / Pi0: build 2-D causal + prefix attention mask."""
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d & pad_2d


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class VLAGemmaHF(nn.Module):
    """
    基于 HuggingFace PaliGemma + Gemma action expert 的 VLA 模型。

    架构参考 openpi/Pi0：
      - PaliGemma (SigLIP vision + Gemma LM) 处理 image + text → past_key_values (KV cache)
      - 独立 Gemma expert 复用 VLM 的 KV cache 作为 prefix，处理 state + noisy_action + timestep → 预测速度场
      - Expert 的 num_layers / num_kv_heads / head_dim 与 VLM language model 对齐，KV cache 直接复用
      - Flow matching 训练（MSE on velocity）
    """

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__()
        self.config = config

        # ---- VLM backbone: PaliGemma ----
        model_name = getattr(config, "gemma_hf_model_name_or_path", None) or "google/paligemma-3b-pt-224"
        if PaliGemmaForConditionalGeneration is None:
            raise ImportError("VLAGemmaHF 需要 transformers 支持 PaliGemmaForConditionalGeneration")

        self.vlm = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
            device_map=None,
            trust_remote_code=True,
        )
        self._model_name = model_name

        # ---- 从 VLM language model 读取关键维度 ----
        vlm_text_cfg = self.vlm.config.text_config
        vlm_hidden = vlm_text_cfg.hidden_size         # e.g. 2048
        vlm_layers = vlm_text_cfg.num_hidden_layers    # e.g. 18
        vlm_kv_heads = vlm_text_cfg.num_key_value_heads  # e.g. 1
        vlm_heads = vlm_text_cfg.num_attention_heads     # e.g. 8
        vlm_head_dim = vlm_text_cfg.head_dim             # e.g. 256

        # ---- Action expert: Gemma ----
        # 关键：层数/kv_heads/head_dim 对齐 VLM，这样 VLM 的 past_key_values 可以直接复用
        if GemmaForCausalLM is None or GemmaConfig is None:
            raise ImportError("VLAGemmaHF 需要 transformers 支持 GemmaForCausalLM")

        num_actions_chunk = int(getattr(config, "num_actions_chunk", 50))
        fixed_action_dim = int(getattr(config, "fixed_action_dim", 32))
        self.num_actions_chunk = num_actions_chunk
        self.fixed_action_dim = fixed_action_dim
        self.action_horizon = num_actions_chunk

        # expert hidden 可以自定义（投影层会适配），但层数/kv结构必须对齐 VLM
        expert_hidden = getattr(config, "action_head_flow_matching_dim", vlm_hidden)
        expert_intermediate = getattr(config, "action_head_flow_matching_intermediate_size", 4096)

        expert_cfg = GemmaConfig(
            hidden_size=expert_hidden,
            num_hidden_layers=vlm_layers,       # 对齐 VLM
            num_attention_heads=vlm_heads,       # 对齐 VLM
            num_key_value_heads=vlm_kv_heads,    # 对齐 VLM
            head_dim=vlm_head_dim,               # 对齐 VLM
            intermediate_size=expert_intermediate,
            vocab_size=1,  # unused
        )
        self.action_expert = GemmaForCausalLM(config=expert_cfg)
        self.action_expert.model.embed_tokens = None  # Pi0-style: no token embedding

        log.info(f"VLAGemmaHF expert: hidden={expert_hidden}, layers={vlm_layers}, "
                 f"heads={vlm_heads}, kv_heads={vlm_kv_heads}, head_dim={vlm_head_dim}")

        # ---- Projection layers (Pi0-style) ----
        self.state_proj = nn.Linear(fixed_action_dim, expert_hidden)
        self.action_in_proj = nn.Linear(fixed_action_dim, expert_hidden)
        self.action_time_mlp_in = nn.Linear(2 * expert_hidden, expert_hidden)
        self.action_time_mlp_out = nn.Linear(expert_hidden, expert_hidden)
        self.action_out_proj = nn.Linear(expert_hidden, fixed_action_dim)

        # ---- Proprio ----
        if config.use_proprio:
            self.proprio_projector = ProprioProjector(
                llm_dim=config.d_model, proprio_dim=fixed_action_dim
            )
        else:
            self.proprio_projector = None

        # ---- compatibility ----
        self.transformer = self.vlm
        self.vision_backbone = None
        self.hidden_size = vlm_hidden

        dtype = self.vlm.dtype
        self.action_expert.to(dtype)
        self.state_proj.to(dtype)
        self.action_in_proj.to(dtype)
        self.action_time_mlp_in.to(dtype)
        self.action_time_mlp_out.to(dtype)
        self.action_out_proj.to(dtype)

    # ===== Flow matching helpers =====

    @torch.no_grad()
    def sample_noisy_actions(self, actions: torch.Tensor):
        B = actions.shape[0]
        device = actions.device
        noise = torch.randn_like(actions)
        t = torch.distributions.Beta(
            torch.tensor(1.5, device=device, dtype=torch.float32),
            torch.tensor(1.0, device=device, dtype=torch.float32),
        ).sample((B,))
        t = (t * 0.999 + 0.001).to(torch.float32)
        t_exp = t[:, None, None]
        x_t = t_exp * noise + (1.0 - t_exp) * actions
        return {"noise": noise, "x_t": x_t.to(actions.dtype), "t": t}

    def embed_suffix(self, state: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """Pi0-style: embed state + noisy_actions + timestep → suffix tokens."""
        dtype = self.state_proj.weight.dtype
        device = x_t.device

        # squeeze state to (B, D) — 兼容 (B, D) / (B, 1, D) / (B, 1, 1, D)
        s = state.to(dtype)
        while s.dim() > 2:
            s = s.squeeze(-2)
        state_tok = self.state_proj(s).unsqueeze(1)  # (B, 1, D)

        time_emb = create_sinusoidal_pos_embedding(t, self.action_in_proj.out_features)
        time_emb = time_emb.to(dtype=dtype, device=device)

        action_emb = self.action_in_proj(x_t.to(dtype))  # (B, T, D)
        time_emb_exp = time_emb[:, None, :].expand_as(action_emb)
        at = self.action_time_mlp_in(torch.cat([action_emb, time_emb_exp], dim=-1))
        at = F.silu(at)
        at = self.action_time_mlp_out(at)

        suffix = torch.cat([state_tok, at], dim=1)  # (B, 1+T, D)
        return suffix

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Convert boolean 2D masks to 4D float masks for transformer."""
        att_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_4d, 0.0, -2.3819763e38)

    def predict_vector_field(
        self,
        prefix_kv,
        prefix_pad_masks: torch.Tensor,
        state: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        pos_offset: torch.Tensor,
    ) -> torch.Tensor:
        """Single denoising step: expert attends to prefix KV cache from VLM.
        与 openpi pi0_pytorch.py denoise_step 一致。
        """
        suffix = self.embed_suffix(state, x_t, t)
        B, L = suffix.shape[:2]
        prefix_len = prefix_pad_masks.shape[1]

        # Pi0-style attention mask construction (same as pi0_pytorch.py L431-L447)
        # suffix pad & att masks
        suffix_pad_masks = torch.ones((B, L), dtype=torch.bool, device=suffix.device)
        suffix_att_masks = torch.zeros((B, L), dtype=torch.int32, device=suffix.device)
        suffix_att_masks[:, 0] = 1  # state token starts new causal block

        # prefix_pad_2d_masks: suffix can attend to all valid prefix positions
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(B, L, prefix_len)

        # suffix causal 2d masks
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        # full mask: (B, suffix_len, prefix_len + suffix_len)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # position ids
        position_ids = pos_offset.view(B, 1) + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # to 4D float mask
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        # expert forward with eager attention (openpi transformers_replace)
        self.action_expert.model.config._attn_implementation = "eager"
        out = self.action_expert.model(
            inputs_embeds=suffix,
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=prefix_kv,
            use_cache=False,
        )
        h = out.last_hidden_state[:, -self.action_horizon:, :]
        return self.action_out_proj(h.to(torch.float32)).to(x_t.dtype)

    # ===== train_for_action.py / FSDP compat =====

    def num_params(self, include_embedding=True, include_inactive_params=True) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def get_connector_parameters():
        return tuple([])

    @staticmethod
    def get_vit_parameters():
        return tuple([])

    @staticmethod
    def get_llm_parameters():
        return tuple(["vlm"])

    @staticmethod
    def get_act_head_parameters():
        return tuple(["action_expert", "state_proj", "action_in_proj",
                       "action_time_mlp_in", "action_time_mlp_out",
                       "action_out_proj"])

    @staticmethod
    def get_proprio_proj_parameters():
        return tuple(["proprio_projector"])

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple([])

    def get_fsdp_basic_modules(self) -> list:
        for m in [self.vlm, getattr(self.vlm, "model", None)]:
            if m is not None:
                mods = getattr(m.__class__, "_no_split_modules", None)
                if mods:
                    return list(mods)
        return ["GemmaDecoderLayer", "SiglipEncoderLayer"]

    def get_fsdp_wrap_policy(self, wrap_strategy=None):
        if wrap_strategy is None:
            return None
        try:
            from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
            basic = self.get_fsdp_basic_modules()
            return partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda module: module.__class__.__name__ in basic,
            )
        except Exception as e:
            log.warning(f"FSDP wrap policy fallback: {e}")
        return None

    def set_activation_checkpointing(self, strategy=None):
        # 关闭 gradient checkpointing，保证 use_cache=True 可用（获取 past_key_values 给 expert）
        # def _enable(module: nn.Module):
        #     if hasattr(module, "gradient_checkpointing_enable"):
        #         try:
        #             module.gradient_checkpointing_enable()
        #         except Exception:
        #             pass
        #     for child in module.children():
        #         _enable(child)
        # _enable(self.vlm)
        pass

    def to_empty(self, device="cpu"):
        return self.to(device)

    def reset_with_pretrained_weights(self, *args, **kwargs):
        return None

    # ===== Forward =====

    def forward(
        self,
        vlm_inputs: Dict[str, Any],
        target_actions: Optional[torch.Tensor] = None,
        action_proprio: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        is_training = self.training or target_actions is not None

        # 从 batch dict 中提取 VLM processor 输出的字段
        proc_inputs = vlm_inputs.get("vlm_inputs", vlm_inputs)
        device = next(self.vlm.parameters()).device
        proc_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in proc_inputs.items()}
        input_ids = proc_inputs["input_ids"]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # use_cache=True 获取 past_key_values，直接给 expert 用
            outputs = self.vlm(
                **proc_inputs,
                output_hidden_states=False,
                return_dict=True,
                use_cache=True,
            )

        if not is_training:
            return outputs

        B = target_actions.shape[0]
        dtype = self.vlm.dtype

        # flow matching
        fm = self.sample_noisy_actions(target_actions)
        noise, x_t, t = fm["noise"], fm["x_t"], fm["t"]
        timesteps = t.unsqueeze(1)

        assert self.config.use_proprio and action_proprio is not None, \
            "flow_matching requires action_proprio"

        # prefix pad masks: 哪些 prefix position 是有效的（非 padding）
        prefix_pad_masks = (input_ids != -1).to(torch.bool)  # (B, S)
        pos_offset = prefix_pad_masks.to(torch.int64).sum(dim=1)

        # 直接使用 VLM 的 past_key_values 作为 expert 的 prefix KV cache
        past_kv = outputs.past_key_values
        past_kv = None
        pred = self.predict_vector_field(
            past_kv, prefix_pad_masks,
            action_proprio.to(dtype), x_t.to(dtype), t.to(dtype),
            pos_offset=pos_offset,
        )
        target = (noise - target_actions).to(dtype)

        return {
            "outputs": outputs,
            "predicted_actions": None,
            "diffusion_target": target,
            "diffusion_pred": pred,
            "diff_timesteps": timesteps,
        }

    @torch.inference_mode()
    def predict_actions(
        self,
        vlm_inputs: Dict[str, Any],
        **kwargs,
    ) -> torch.Tensor:
        proc_inputs = vlm_inputs.get("vlm_inputs", vlm_inputs)
        device = next(self.vlm.parameters()).device
        proc_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in proc_inputs.items()}
        input_ids = proc_inputs["input_ids"]
        pos_offset = (input_ids != -1).to(torch.int64).sum(dim=1)

        # VLM forward once, cache KV
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.vlm(
                **proc_inputs,
                output_hidden_states=False,
                return_dict=True,
                use_cache=True,
            )
        past_kv = outputs.past_key_values
        prefix_pad_masks = (input_ids != -1).to(torch.bool)

        B = input_ids.shape[0]
        dtype = self.vlm.dtype

        # Euler ODE solver
        steps = getattr(self.config, "num_diffusion_inference_steps", 10)
        dt = -1.0 / float(steps)
        x = torch.randn(
            (B, self.num_actions_chunk, self.fixed_action_dim), device=device, dtype=dtype
        )
        t_float = 1.0

        assert self.config.use_proprio, "flow_matching requires use_proprio"
        state = kwargs.get("action_proprio")
        assert state is not None, "action_proprio required for inference"

        for _ in range(steps):
            t = torch.full((B,), t_float, device=device, dtype=dtype)
            v = self.predict_vector_field(
                past_kv, prefix_pad_masks, state.to(dtype), x, t, pos_offset=pos_offset
            )
            x = x + dt * v
            t_float += dt
        return x


def build_vla_gemma_hf(config: ModelConfig, **kwargs) -> VLAGemmaHF:
    return VLAGemmaHF(config, **kwargs)
