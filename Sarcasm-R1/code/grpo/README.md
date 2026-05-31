# GRPO Training Module

Train a small language model to perform structured sarcasm reasoning using GRPO.

## Files

| File | Description |
|------|-------------|
| `train.py` | Main training script |
| `config.yaml` | Hyperparameter configuration |
| `deepspeed_zero2.yaml` | DeepSpeed ZeRO-2 distributed training config |
| `rewards.py` | Reward functions (accuracy, format, dimension coverage) |
| `prompts.py` | System and user prompt templates |

## Training

```bash
# Single GPU
accelerate launch train.py --config config.yaml

# Multi-GPU with DeepSpeed ZeRO-2
accelerate launch --config_file=deepspeed_zero2.yaml \
    --num_processes 7 \
    train.py --config config.yaml
```

## Reward Functions

Three reward functions guide training:

1. **format_reasoning_reward**: Rewards structured reasoning with proper length and `{answer}` format
2. **accuracy_reward_log**: Logs and rewards correct sarcasm judgments (+10/-5)
3. **dimension_coverage_reward** (via accuracy_reward_log): Rewards covering Rhetoric/Context/Emotion dimensions

## Configuration

Key parameters in `config.yaml`:

- `model_name_or_path`: Base model (e.g., `Qwen/Qwen2.5-7B-Instruct`)
- `num_generations`: Group size for GRPO (default: 4)
- `learning_rate`: 3e-6 with cosine schedule
- `max_prompt_length` / `max_completion_length`: 1024 tokens each
