from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

from torch.utils.data import IterableDataset  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.data import Data
from envs.logic_circuit.env import LogicCircuitEnv
from training.common import load_system_prompt


class TrainIterableDataset(IterableDataset):
    def __init__(self, iterator: Iterator[Dict[str, Any]]):
        super().__init__()
        self._iterator = iterator

    def __iter__(self):
        return self._iterator


def iter_train(
    *,
    env: LogicCircuitEnv,
    rows: List[Data],
    system_prompt: str,
    seed: int = 0,
) -> Iterator[Dict[str, Any]]:
    rng = random.Random(seed)
    while True:
        it = rows[rng.randint(0, len(rows) - 1)]
        yield {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": it.question},
            ],
            "reference": it.to_json(),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="GRPO-only training with verifier reward.")
    parser.add_argument("--init-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train", default="datasets/train_full.jsonl")
    parser.add_argument("--system-prompt", default="training/SYSTEM_PROMPT.txt")
    parser.add_argument("--output-dir", default="artifacts/model_grpo")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from unsloth import FastLanguageModel  # type: ignore
    except Exception as e:
        raise SystemExit(f"unsloth import failed: {e}")

    try:
        from trl import GRPOConfig, GRPOTrainer  # type: ignore
    except Exception as e:
        raise SystemExit(f"trl.GRPO import failed: {e}")

    system_prompt = load_system_prompt(args.system_prompt)
    env = LogicCircuitEnv()
    rows = Data.from_jsonl_file(args.train)

    train_dataset = TrainIterableDataset(iter_train(env=env, rows=rows, system_prompt=system_prompt, seed=args.seed))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.init_model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    def correctness_reward_func(prompts: List[Any], completions: List[str], reference: List[Dict[str, Any]], **kwargs):
        rewards: List[float] = []
        for ref, completion in zip(reference, completions):
            data = Data.from_json_dict(ref)
            ok = env.verify(data, completion)
            rewards.append(1.0 if ok else 0.0)
        return rewards

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        max_steps=args.steps,
        logging_steps=10,
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[correctness_reward_func],
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    summary = {
        "init_model": args.init_model,
        "train": args.train,
        "steps": args.steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
    }
    Path(args.output_dir, "grpo_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved GRPO model to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
