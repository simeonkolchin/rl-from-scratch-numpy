# HW3 Report: RL-PLUS on logic_circuit

## Setup
- Model: `Qwen/Qwen2.5-0.5B-Instruct` (QLoRA 4bit)
- System prompt: `training/SYSTEM_PROMPT.txt`
- Fixed sampling for all comparisons:
  - `temperature=1.0`
  - `top_p=0.95`
  - `max_tokens=256`

## Data splits
- `train_full.jsonl`: 4000
- `val_full.jsonl`: 300
- `hard_train.jsonl`: 512 (baseline pass@128=0)
- `hard_val.jsonl`: 128 (baseline pass@128=0)
- `gold_train.jsonl`: algorithmic CoT + verified answer

## Protocol
1. Baseline eval on `val_full` and `hard_val`.
2. GRPO-only training/eval.
3. SFT-only training/eval (optional dedicated table).
4. SFT->GRPO training/eval.
5. RL-PLUS training/eval.

## Results placeholders

### Main pass@k (val full)
| Model | pass@1 | pass@4 | pass@8 | pass@16 | pass@32 | pass@64 | pass@128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline |  |  |  |  |  |  |  |
| GRPO-only |  |  |  |  |  |  |  |
| SFT->GRPO |  |  |  |  |  |  |  |
| RL-PLUS |  |  |  |  |  |  |  |

### Main pass@k (Hard-val)
| Model | pass@1 | pass@4 | pass@8 | pass@16 | pass@32 | pass@64 | pass@128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline |  |  |  |  |  |  |  |
| GRPO-only |  |  |  |  |  |  |  |
| SFT->GRPO |  |  |  |  |  |  |  |
| RL-PLUS |  |  |  |  |  |  |  |

### Zero-breaking check
- Baseline on Hard-val pass@128: `0`
- Best trained model on Hard-val pass@128: `>0` / `not achieved`

### Dynamics
- Reward curve: `reports/figures/reward_curve.png`
- Entropy curve: `reports/figures/entropy_curve.png`
- Length curve: `reports/figures/length_curve.png`

### Generation length stats
| Model | Dataset | Mean len | Median len |
|---|---|---:|---:|
| baseline | val_full |  |  |
| baseline | hard_val |  |  |
| GRPO-only | val_full |  |  |
| GRPO-only | hard_val |  |  |
| SFT->GRPO | val_full |  |  |
| SFT->GRPO | hard_val |  |  |
| RL-PLUS | val_full |  |  |
| RL-PLUS | hard_val |  |  |

## Trade-off quality vs diversity
- Discuss whether `val_full` pass@k drops after RL.
- Analyze connection to entropy/length/off-policy ratio.
