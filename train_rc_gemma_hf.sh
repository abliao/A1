# ============================================
# Load personal environment configuration
# ============================================
if [ -f "$PWD/.env.personal" ]; then
  echo "[env] Loading .env.personal"
  source "$PWD/.env.personal"
fi
# ============================================
# Activate Conda environment
# ============================================
if [ -n "$CONDA_ROOT" ] && [ -n "$CONDA_ENV" ]; then
  echo "[conda] Activating environment from $CONDA_ROOT: $CONDA_ENV"
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

# Training configuration
export dataset_name=vla_dataset_rc
vla_config_path="rc_open_the_drawer_gemma_hf.yaml"
exp_name="a1_rc_open_the_drawer_gemma_hf"
save_folder="./model/checkpoints/$exp_name"

# Automatically set nproc_per_node based on visible GPU count
if [ -n "${CUDA_VISIBLE_DEVICES-}" ]; then
  IFS=',' read -ra DEV_ARR <<< "${CUDA_VISIBLE_DEVICES}"
  nproc_per_node=${#DEV_ARR[@]}
else
  nproc_per_node=$(nvidia-smi -L | wc -l)
fi
BATCH_PER_GPU=8
STATE_MASK_PROB="0.0"
global_batch_size=$((nproc_per_node * BATCH_PER_GPU))

# Launch training
# PaliGemma-3B: d_model=2048, n_layers=18, n_heads=8, n_kv_heads=1
# 使用 qwen2_7b 作为 base config（仅用于提供 ModelConfig 基础字段），
# 实际 VLM 由 gemma_hf_model_name_or_path 指定的 PaliGemma 加载。
torchrun \
  --nproc-per-node=$nproc_per_node \
  --rdzv-endpoint=localhost:13301 \
  launch_scripts/train_vla.py \
  qwen2_7b \
  save_folder=$save_folder \
  --action_head "flow_matching" \
  --seq_len 600 \
  --state_mask_prob "${STATE_MASK_PROB}" \
  --device_train_microbatch_size $BATCH_PER_GPU \
  --global_batch_size $global_batch_size \
  --dataset $dataset_name \
  --ft_llm \
  --llm_learning_rate 5e-6 \
  --action_head_learning_rate 5e-5 \
  --warmup_steps 2000 \
  --freeze_steps 1000 \
  --save_interval_unsharded 1000 \
  --save_interval 1000 \
  --train_steps 50000 \
  --vla_config_path $vla_config_path \
  --fsdp_mode fsdp2 \
  --wandb_entity $WANDB_ENTITY \
  --wandb_project $WANDB_PROJECT \
  --wandb_run_name $exp_name \
  --save_overwrite \
  --log_interval 50 \
  --num_workers 4
