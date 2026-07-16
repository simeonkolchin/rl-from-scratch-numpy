#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
TEMP=1.0
TOP_P=0.95
MAX_TOKENS=256

python training/make_datasets.py --train-size 4000 --val-size 300 --seed 42
python training/make_gold.py --train datasets/train_full.jsonl --out datasets/gold_train.jsonl --limit 4000

python training/make_hard_subsets.py \
  --train datasets/train_full.jsonl \
  --val datasets/val_full.jsonl \
  --model "$MODEL" \
  --n-screen 16 \
  --n-final 128 \
  --hard-train-size 512 \
  --hard-val-size 128 \
  --temperature $TEMP \
  --top-p $TOP_P \
  --max-tokens $MAX_TOKENS

python training/eval_passk.py --model "$MODEL" --dataset datasets/val_full.jsonl --n 128 --out artifacts/baseline_full.json --temperature $TEMP --top-p $TOP_P --max-tokens $MAX_TOKENS
python training/eval_passk.py --model "$MODEL" --dataset datasets/hard_val.jsonl --n 128 --out artifacts/baseline_hard.json --temperature $TEMP --top-p $TOP_P --max-tokens $MAX_TOKENS

python training/train_grpo.py --init-model "$MODEL" --train datasets/train_full.jsonl --steps 300 --output-dir artifacts/model_grpo
python training/eval_passk.py --model artifacts/model_grpo --dataset datasets/val_full.jsonl --n 128 --out artifacts/grpo_full.json --temperature $TEMP --top-p $TOP_P --max-tokens $MAX_TOKENS
python training/eval_passk.py --model artifacts/model_grpo --dataset datasets/hard_val.jsonl --n 128 --out artifacts/grpo_hard.json --temperature $TEMP --top-p $TOP_P --max-tokens $MAX_TOKENS

python training/train_sft.py --model "$MODEL" --train-gold datasets/gold_train.jsonl --output-dir artifacts/model_sft --epochs 1 --lr 2e-4 --lora-r 16
python training/train_grpo.py --init-model artifacts/model_sft --train datasets/train_full.jsonl --steps 200 --output-dir artifacts/model_sft_grpo
python training/eval_passk.py --model artifacts/model_sft_grpo --dataset datasets/val_full.jsonl --n 128 --out artifacts/sftgrpo_full.json --temperature $TEMP --top-p $TOP_P --max-tokens $MAX_TOKENS
python training/eval_passk.py --model artifacts/model_sft_grpo --dataset datasets/hard_val.jsonl --n 128 --out artifacts/sftgrpo_hard.json --temperature $TEMP --top-p $TOP_P --max-tokens $MAX_TOKENS

python training/train_rlplus.py --init-model "$MODEL" --train datasets/train_full.jsonl --gold datasets/gold_train.jsonl --steps 300 --lambda-off 0.5 --off-min-clip 0.2 --off-max-clip 5.0 --output-dir artifacts/model_rlplus
python training/eval_passk.py --model artifacts/model_rlplus --dataset datasets/val_full.jsonl --n 128 --out artifacts/rlplus_full.json --temperature $TEMP --top-p $TOP_P --max-tokens $MAX_TOKENS
python training/eval_passk.py --model artifacts/model_rlplus --dataset datasets/hard_val.jsonl --n 128 --out artifacts/rlplus_hard.json --temperature $TEMP --top-p $TOP_P --max-tokens $MAX_TOKENS

python training/plot_report.py \
  --baseline-full artifacts/baseline_full.json \
  --baseline-hard artifacts/baseline_hard.json \
  --grpo-full artifacts/grpo_full.json \
  --grpo-hard artifacts/grpo_hard.json \
  --sftgrpo-full artifacts/sftgrpo_full.json \
  --sftgrpo-hard artifacts/sftgrpo_hard.json \
  --rlplus-full artifacts/rlplus_full.json \
  --rlplus-hard artifacts/rlplus_hard.json

echo "Experiment pipeline completed."
