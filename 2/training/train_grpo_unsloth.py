from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from torch.utils.data import IterableDataset  # type: ignore

from base.data import Data
from envs.logic_circuit.env import LogicCircuitEnv


def load_system_prompt(path: str = "training/SYSTEM_PROMPT.txt") -> str:
    return Path(path).read_text(encoding="utf-8").strip() + "\n"


def iter_train(
    *,
    env: LogicCircuitEnv,
    system_prompt: str,
    seed: int = 0,
    max_attempts: int = 200,
    difficulties: Optional[List[int]] = None,
) -> Iterator[Dict[str, Any]]:
    import random

    rng = random.Random(seed)
    if not difficulties:
        difficulties = list(range(1, 11))

    while True:
        d = rng.choice(difficulties)
        sample_seed = rng.randint(0, 2**31 - 1)
        data = env.generate(num_of_questions=1, max_attempts=max_attempts, difficulty=d, seed=sample_seed)[0]
        yield {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data.question},
            ],
            "reference": data.to_json(),
        }


class TrainIterableDataset(IterableDataset):
    def __init__(self, iterator: Iterator[Dict[str, Any]]):
        super().__init__()
        self._iterator = iterator

    def __iter__(self):
        return self._iterator


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Qwen2.5 with GRPO on logic_circuit using unsloth.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", default="artifacts/model_grpo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    try:
        from unsloth import FastLanguageModel  # type: ignore
    except Exception as e:
        raise SystemExit(
            "unsloth is not installed (or failed to import). "
            "Install deps from training/requirements-train.txt in a GPU environment.\n"
            f"Import error: {e}"
        )

    try:
        from trl import GRPOConfig, GRPOTrainer  # type: ignore
    except Exception as e:
        raise SystemExit(
            "trl GRPOTrainer not available. Install deps from training/requirements-train.txt.\n"
            f"Import error: {e}"
        )

    system_prompt = load_system_prompt()
    env = LogicCircuitEnv()

    train_dataset = TrainIterableDataset(
        iter_train(env=env, system_prompt=system_prompt, seed=int(args.seed))
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    def correctness_reward_func(prompts: List[Any], completions: List[str], reference: List[Dict[str, Any]], **kwargs):
        rewards: List[float] = []
        for ref, completion in zip(reference, completions):
            data = Data.from_json_dict(ref)
            ok = env.verify(data, completion)
            rewards.append(1.0 if ok else 0.0)
        return rewards

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=float(args.lr),
        per_device_train_batch_size=int(args.batch_size),
        max_steps=int(args.steps),
        logging_steps=10,
        seed=int(args.seed),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[correctness_reward_func],
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved model to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
