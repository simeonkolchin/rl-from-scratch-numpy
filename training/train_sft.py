from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import Dataset  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments  # type: ignore
from peft import LoraConfig, get_peft_model  # type: ignore

from training.common import load_system_prompt, read_jsonl


def build_text(system_prompt: str, prompt: str, completion: str) -> str:
    return (
        f"system\n{system_prompt}\n"
        f"user\n{prompt}\n"
        f"assistant\n{completion}"
    )


def tokenize_fn(tokenizer, max_length: int):
    def _fn(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=max_length)
        out["labels"] = out["input_ids"].copy()
        return out

    return _fn


def main() -> int:
    parser = argparse.ArgumentParser(description="SFT on gold trajectories.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-gold", default="datasets/gold_train.jsonl")
    parser.add_argument("--system-prompt", default="training/SYSTEM_PROMPT.txt")
    parser.add_argument("--output-dir", default="artifacts/model_sft")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(args.train_gold)
    system_prompt = load_system_prompt(args.system_prompt)

    texts: List[str] = []
    for r in rows:
        texts.append(build_text(system_prompt=system_prompt, prompt=r["prompt"], completion=r["gold_completion"]))

    ds = Dataset.from_dict({"text": texts})

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=max(16, args.lora_r),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_cfg)

    tokenized = ds.map(tokenize_fn(tokenizer, args.max_length), batched=True, remove_columns=["text"])

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(model=model, args=train_args, train_dataset=tokenized, tokenizer=tokenizer)
    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    summary = {
        "model": args.model,
        "train_gold": args.train_gold,
        "output_dir": args.output_dir,
        "rows": len(rows),
    }
    Path(args.output_dir, "sft_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved SFT model to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
