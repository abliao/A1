# Copyright (c) 2025.
# VLA 使用 Transformers 的 Qwen3-VL + AutoProcessor 实现，不依赖 Molmo 的 preprocessor/vision 代码。
# 参考 starVLA：/data/zhangkaidong/starVLA
#
# 训练管线保持与原 VLATrainer 兼容：
# - dataloader 通过 HFQwenVLCollatorForAction 产出 batch dict（含 input_ids/attention_mask/images/action/...）
# - VLATrainer 仍调用 model.forward(input_ids=..., images=..., target_actions=...) 并计算 action_loss

from __future__ import annotations

import logging
from functools import partial
from typing import Any, List, Optional, Dict

import torch
import torch.nn as nn

from a1.config import ModelConfig
from a1.vla.action_heads import FlowMatchingActionHeadQwen3, FlowMatchingActionHead, HiddenStatesToKVCache
from a1.vla.affordvla import ProprioProjector
import torch.nn.functional as F

log = logging.getLogger(__name__)

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import AutoProcessor
except ImportError:
    AutoProcessor = None

IGNORE_INDEX = -100


class VLAQwen3HF(nn.Module):
    """
    基于 HuggingFace Qwen3-VL（或 Qwen2.5-VL）的 VLA 模型（HF 后端）。

    目标：只“换模型 + 换 dataloader”，训练流程仍走原 VLATrainer。
    - 输入：由 HFQwenVLCollatorForAction 生成 batch dict：
      - input_ids: (B, L)
      - attention_mask: (B, L)
      - images: dict（processor 输出中除 input_ids/attention_mask 外的视觉张量，如 pixel_values/image_grid_thw/...）
      - action: (B, T, A)
      - action_pad_mask: (B, T, A) bool
      - proprio/proprio_token_idx: 可选/占位
    - 输出：返回与 AffordVLA 相同键：
      {"outputs": ..., "predicted_actions": ..., "diffusion_target": None, "diffusion_pred": None, "diff_timesteps": None}
    """

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__()
        self.config = config
        model_name = getattr(config, "qwen3_hf_model_name_or_path", None) or "Qwen/Qwen3-VL-4B-Instruct"
        use_qwen3 = "qwen3" in model_name.lower() or "Qwen3" in model_name

        # if AutoProcessor is None:
        #     raise ImportError("VLAQwen3HF 需要 transformers，请安装: pip install transformers")

        # self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        # self.processor.tokenizer.padding_side = "left"

        if use_qwen3 and Qwen3VLForConditionalGeneration is not None:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                attn_implementation="sdpa",
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
                device_map=None,
                trust_remote_code=True,
            )
        elif Qwen2_5_VLForConditionalGeneration is not None:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                attn_implementation="sdpa",
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
                device_map=None,
                trust_remote_code=True,
            )
        else:
            raise ImportError("需要 transformers 支持 Qwen3VLForConditionalGeneration 或 Qwen2_5_VLForConditionalGeneration")

        self._model_name = model_name
        # 兼容 train_for_action.py / optim.py 里对属性名的假设
        self.transformer = self.model
        self.vision_backbone = None

        # 简单的动作回归头：用最后一个 token 的 hidden state 预测整个动作 chunk
        # 配置完全参考 AffordVLA：num_actions_chunk / fixed_action_dim
        num_actions_chunk = int(getattr(config, "num_actions_chunk", 8))
        fixed_action_dim = int(
            getattr(config, "fixed_action_dim", getattr(config, "action_dim", 7))
        )
        self.num_actions_chunk = num_actions_chunk
        self.fixed_action_dim = fixed_action_dim

        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.model.config, "text_config"):
            hidden_size = getattr(self.model.config.text_config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(config, "d_model", 1024)
        self.hidden_size = int(hidden_size)

        qwen_hidden = getattr(self.config, 'action_head_flow_matching_dim', 896)
        qwen_num_layers = getattr(self.config, 'action_head_flow_matching_layers', self.config.n_layers)
        qwen_num_heads = getattr(self.config, 'action_head_flow_matching_heads', 8)
        qwen_num_kv_heads = getattr(
            self.config,
            'action_head_flow_matching_kv_heads',
            self.config.n_kv_heads if getattr(self.config, 'n_kv_heads', None) is not None else self.config.n_heads,
        )
        qwen_num_kv_heads = qwen_num_kv_heads or qwen_num_heads
        if use_qwen3:
            self.action_head = FlowMatchingActionHeadQwen3(
                    llm_dim=self.config.d_model,
                    action_dim=config.fixed_action_dim,
                    proprio_dim=config.fixed_action_dim,
                    horizon=config.num_actions_chunk,
                    qwen3_hidden_size=qwen_hidden,
                    qwen3_num_layers=qwen_num_layers,
                    qwen3_num_heads=qwen_num_heads,
                    qwen3_intermediate_size=getattr(self.config, 'action_head_flow_matching_intermediate_size', 2048),
                    qwen3_num_kv_heads=qwen_num_kv_heads,
                )
            self.hidden_to_kv = HiddenStatesToKVCache(
                hidden_size=self.hidden_size,
                num_layers=qwen_num_layers,
                num_kv_heads=qwen_num_kv_heads,
                head_dim=qwen_hidden // qwen_num_heads,
            )
        else:
            self.action_head = FlowMatchingActionHead(
                llm_dim=self.config.d_model,
                action_dim=config.fixed_action_dim,  
                proprio_dim=config.fixed_action_dim,
                horizon=config.num_actions_chunk,
                qwen2_hidden_size=qwen_hidden,
                qwen2_num_layers=qwen_num_layers,
                qwen2_num_heads=qwen_num_heads,
                qwen2_intermediate_size=getattr(self.config, 'action_head_flow_matching_intermediate_size', 2048),
                qwen2_num_kv_heads=qwen_num_kv_heads,
            )
            self.hidden_to_kv = HiddenStatesToKVCache(
                hidden_size=self.hidden_size,
                num_layers=qwen_num_layers,
                num_kv_heads=qwen_num_kv_heads,
                head_dim=qwen_hidden // qwen_num_heads,
            )
        if config.use_proprio:
            if config.proprio_dim != config.action_dim:
                print(f"config.proprio_dim {config.proprio_dim} does not match config.action_dim {config.action_dim} for AffordVLA")
            self.proprio_projector = ProprioProjector(llm_dim=config.d_model,proprio_dim=config.fixed_action_dim)
        else:
            self.proprio_projector = None
        self.head_dtype = self.model.dtype
        self.action_head.to(self.head_dtype)
        self.hidden_to_kv.to(self.head_dtype)

    # ===== train_for_action.py / FSDP 兼容接口 =====
    def num_params(self, include_embedding: bool = True, include_inactive_params: bool = True) -> int:
        del include_embedding, include_inactive_params
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def get_connector_parameters():
        return tuple(
            [
            ]
        )

    @staticmethod
    def get_vit_parameters():
        return tuple(
            [
            ]
        )

    @staticmethod
    def get_llm_parameters():
        return tuple(
            [
                "model",
            ]
        )

    @staticmethod
    def get_act_head_parameters():
        return tuple[str, ...](["action_head", "hidden_to_kv"])
    
    @staticmethod
    def get_proprio_proj_parameters():
        return tuple(["proprio_projector",])

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple(
            [
            ]
        )

    def get_fsdp_basic_modules(self) -> list:
        """返回 FSDP2 fully_shard 需要的 basic_modules 列表（按 lingbot-vla）"""
        basic_modules = []
        for m in [self.model, getattr(self.model, "model", None)]:
            if m is not None:
                mods = getattr(m.__class__, "_no_split_modules", None)
                if mods:
                    return list(mods)
        return [
            "Qwen3VLTextDecoderLayer",
            "Qwen3VLVisionBlock",
            "Qwen2_5_VLDecoderLayer",
            "Qwen2VLDecoderLayer",
        ]

    def get_fsdp_wrap_policy(self, wrap_strategy=None):
        # 按 lingbot-vla 策略：用 HF model._no_split_modules + lambda_auto_wrap_policy
        if wrap_strategy is None:
            return None
        try:
            from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

            basic_modules = self.get_fsdp_basic_modules()
            for m in [self.model, getattr(self.model, "model", None)]:
                if m is not None:
                    mods = getattr(m.__class__, "_no_split_modules", None)
                    if mods:
                        basic_modules = list(mods)
                        break
            wrap_policy = partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda module: module.__class__.__name__ in basic_modules,
            )
            return wrap_policy
        except Exception as e:
            log.warning(f"FSDP lambda_auto_wrap_policy fallback to root wrap: {e}")
        return None

    def set_activation_checkpointing(self, strategy=None):
        # return
        # 仅对主 VLM 开启 gradient checkpointing。action_head 会动态修改 config.num_hidden_layers，
        # 与 checkpoint backward 不兼容，易导致 size mismatch（如 q_proj input 维度错误）。
        
        def _enable(module: nn.Module) -> None:
            if hasattr(module, "gradient_checkpointing_enable"):
                try:
                    module.gradient_checkpointing_enable()
                except Exception:
                    pass
            for child in module.children():
                _enable(child)

        _enable(self.model)

    def to_empty(self, device="cpu"):
        # 兼容 train_for_action 在 meta init_device 下的调用
        return self.to(device)

    def reset_with_pretrained_weights(self, *args, **kwargs):
        # HF 模型权重由 from_pretrained() 负责，这里保持 no-op 即可
        return None

    def forward(
        self,
        vlm_inputs: Dict[str, Any],
        target_actions: Optional[torch.Tensor] = None,
        action_proprio: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        is_training = self.training or target_actions is not None
    
        input_ids = vlm_inputs["input_ids"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **vlm_inputs, 
                output_hidden_states=True, 
                return_dict=True,
                use_cache=False
            )  
        if not is_training:
            return outputs
        
        B = target_actions.shape[0]
        device = target_actions.device
        dtype = self.model.dtype
        fm = self.action_head.sample_noisy_actions(target_actions)
        noise, x_t, t = fm['noise'], fm['x_t'], fm['t']
        timesteps = t.unsqueeze(1)

        assert self.config.use_proprio and action_proprio is not None, "flow_matching requires action_proprio"
        pos_offset = (input_ids != -1).to(torch.int64).sum(dim=1)
        print('input_ids',input_ids.shape)
        pkv = outputs.past_key_values
        if pkv is None and self.hidden_to_kv is not None:
            hs_all = outputs.hidden_states
            pkv = self.hidden_to_kv(tuple(h.to(dtype) for h in hs_all))
        
        pred = self.action_head.predict_vector_field(
            pkv,
            action_proprio.to(dtype),
            x_t.to(dtype),
            t.to(dtype),
            pos_offset=pos_offset,
        )
        target = (noise - target_actions).to(dtype)
        # 维持 FM 路径的 dtype（AMD 上为 fp32），交由后续 loss 在 fp32 计算
        predicted_actions = None

        return {   
            'outputs': outputs,
            'predicted_actions': predicted_actions,  
            'diffusion_target': target,
            'diffusion_pred': pred,
            'diff_timesteps': timesteps,
        }

    @torch.inference_mode()
    def predict_actions(
        self,
        vlm_inputs: Dict[str, Any],
        **kwargs,
    ) -> torch.Tensor:
        input_ids = vlm_inputs["input_ids"]
        pos_offset = (input_ids != -1).to(torch.int64).sum(dim=1)
        out = self.forward(vlm_inputs=vlm_inputs)
        pkv = out.past_key_values
        if pkv is None and self.hidden_to_kv is not None:
            hs_all = getattr(out, "hidden_states", None)
            if hs_all is None:
                hs_all = (out.last_hidden_state,)
            pkv = self.hidden_to_kv(tuple(h for h in hs_all))

        
        # Euler steps with expert only
        device = input_ids.device
        dtype = out.last_hidden_state.dtype
        B = input_ids.shape[0]
        steps = getattr(self.config, 'num_diffusion_inference_steps', 10)
        dt = -1.0 / float(steps)
        x = torch.randn((B, self.config.num_actions_chunk, self.config.fixed_action_dim), device=device, dtype=dtype)
        t_float = 1.0
        # use proprio as state if configured
        assert self.config.use_proprio, "flow_matching requires use_proprio=True for state token"
        # the caller must pass action_proprio via kwargs in predict stage
        state = kwargs.get('action_proprio', None)
        assert state is not None, "action_proprio is required for flow_matching inference"
        # 基于 prefix_pad_masks 计算每个样本的有效前缀长度，避免 padding 干扰位置编码
        if 'prefix_pad_masks' in kwargs and kwargs['prefix_pad_masks'] is not None:
            ppm = kwargs['prefix_pad_masks']  # bool[B, P]
            pos_offset = ppm.to(torch.int64).sum(dim=1)  # (B,)

        for _ in range(steps):
            t = torch.full((B,), t_float, device=device, dtype=dtype)
            v = self.action_head.predict_vector_field(pkv, state, x, t, pos_offset=pos_offset)
            x = x + dt * v
            t_float += dt
        predicted_actions = x
        
        return predicted_actions


def build_vla_qwen3_hf(config: ModelConfig, **kwargs) -> VLAQwen3HF:
    return VLAQwen3HF(config, **kwargs)
