# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A1 is a Vision-Language-Action (VLA) model for robotic manipulation, built on the Molmo (OLMo-based) vision-language backbone with Qwen2-7B as the LLM. It features early exit / truncated vision mechanisms for adaptive efficiency. Paper: arXiv 2604.05672.

## Environment Setup

```bash
conda create -n a1 python=3.10 && conda activate a1
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e .[all]
pip install --no-deps --force-reinstall git+https://github.com/moojink/dlimp_openvla
pip install -r requirements.txt
cp .env.example .env.personal  # then edit with your paths/keys
source .env.personal
```

**Special setup for pretraining:** LeRobot requires patching: `cp a1/data/vla/lerobot_datasets_replace.py <CONDA_ENV>/lib/python3.10/site-packages/lerobot/datasets/lerobot_dataset.py`

**RoboMIND preprocessing:** `bash scripts/robomind_build_index.sh`

## Common Commands

### Training (all scripts auto-load `.env.personal` and activate conda)
```bash
bash train_libero.sh          # LIBERO fine-tuning
bash train_vlabench.sh        # VLABench fine-tuning
bash train_rc.sh              # RoboChallenge fine-tuning (edit vla_config_path for task)
bash scripts/slurms/pretrain.sh           # Pretraining (single node)
bash scripts/slurms/submit_job.sh         # Pretraining (Slurm multi-node)
```

Training uses `torchrun` with FSDP. Entry point: `launch_scripts/train_vla.py`. The model name argument is `qwen2_7b`.

### Deployment & Inference
```bash
bash deploy/deploy.sh --weight <checkpoint_path> --port 8000
```
Starts a FastAPI server with `/health`, `/inference`, `/infer_batch` endpoints.

### Evaluation
```bash
bash eval_libero.sh           # Standard LIBERO eval
bash eval_libero_exit.sh      # Early exit LIBERO eval
# VLABench: start deploy server in terminal 1, then:
python robot_experiments/vlabench/eval_client.py
# RoboChallenge mock:
python robot_experiments/RoboChallengeInference/run_task.py --task_name open_the_drawer --test_type mock --url http://localhost:8000
```

### Linting / Formatting (dev extras)
```bash
ruff check .
black --check .
isort --check .
pytest
```

## Architecture

### Core Model Pipeline
`a1/model.py` (Molmo backbone) → `a1/vla/affordvla.py` (AffordVLA wrapper) → `a1/vla/action_heads.py` (action prediction)

- **Vision encoder**: CLIP/SigLIP/DinoV2 options (`a1/image_vit.py`), selected via `--vision_backbone`
- **LLM**: Qwen2-7B processes vision tokens + text instruction
- **Action heads**: FlowMatching (default), DiffusionTransformer, Diffusion, L1Regression — selected via `--action_head`
- **Early exit**: Truncated vision processing for efficiency (see `eval_libero_early_exit.py`)

### Data Pipeline
`a1/data/A1_datasets.py` + `a1/data/vla/` (per-format loaders) → `a1/data/collator.py` (MMCollator) → `a1/data/model_preprocessor.py`

Supports multiple dataset formats: RLDS, LeRobot, AgiBot, RoboMIND, RoboChallenge. Datasets are mixed with weighted sampling.

### Configuration System
YAML configs in `configs/` with three layers:
- `configs/models/` — action head architecture (action dim, chunk size, token mapping)
- `configs/datasets/` — data sources and normalization
- `configs/experiments/` — combines model + dataset config via `model_config` and `dataset_config` keys

Config loader (`a1/vla/config_loader.py`) auto-discovers configs from both `configs/` and legacy `launch_scripts/` locations.

### Training System
- Entry: `launch_scripts/train_vla.py` → `a1/train.py`
- Distributed: FSDP with activation checkpointing
- Per-component learning rates: LLM, ViT, connector, action head each configurable separately
- Monitoring: Weights & Biases integration
- Checkpoints: supports both sharded and unsharded formats, auto-resume

### Key Directories
- `a1/` — core model, training, data, eval code
- `configs/` — YAML experiment/model/dataset configs
- `launch_scripts/` — training entry points
- `deploy/` — FastAPI inference server
- `robot_experiments/` — evaluation on LIBERO, VLABench, RoboChallenge (with git submodules)
- `scripts/` — utility and Slurm scripts
- `model/` and `data/` — symlinks to checkpoint and dataset storage
