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
from training.common import load_system_prompt, read_jsonl


class RLPlusIterableDataset(IterableDataset):
    def __init__(self, iterator: Iterator[Dict[str, Any]]):
        super().__init__()
        self._iterator = iterator

    def __iter__(self):
        return self._iterator


def iter_mixed(
    *,
    train_rows: List[Data],
    gold_rows: List[Dict[str, Any]],
    system_prompt: str,
    seed: int,
    off_policy_ratio: float,
) -> Iterator[Dict[str, Any]]:
    rng = random.Random(seed)
    gold_by_id = {r["id"]: r for r in gold_rows}
    train_ids = [x.id for x in train_rows]

    while True:
        it = train_rows[rng.randint(0, len(train_rows) - 1)]
        use_off = rng.random() < off_policy_ratio and it.id in gold_by_id
        if use_off:
            g = gold_by_id[it.id]
            prefix = g["gold_completion"]
            prompt = f"{it.question}\n\nPrefix (teacher trajectory):\n{prefix}\nContinue from this prefix."
            off_policy = True
        else:
            prompt = it.question
            off_policy = False

        yield {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "reference": it.to_json(),
            "off_policy": off_policy,
            "teacher": gold_by_id.get(it.id, {}).get("gold_completion", ""),
        }


def _exploration_bonus(text: str) -> float:
    tokens = len(text.split())
    if tokens < 16:
        return -0.1
    if tokens > 256:
        return -0.1
    unique = len(set(text.split()))
    return min(0.2, unique / max(1.0, tokens) * 0.5)


def _teacher_match_score(completion: str, teacher: str) -> float:
    if not teacher:
        return 0.0
    teacher_tokens = teacher.split()
    pred_tokens = completion.split()
    if not teacher_tokens or not pred_tokens:
        return 0.0
    m = 0
    for a, b in zip(pred_tokens, teacher_tokens):
        if a == b:
            m += 1
        else:
            break
    return m / max(1, len(teacher_tokens))


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight single-GPU RL-PLUS style training.")
    parser.add_argument("--init-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train", default="datasets/train_full.jsonl")
    parser.add_argument("--gold", default="datasets/gold_train.jsonl")
    parser.add_argument("--system-prompt", default="training/SYSTEM_PROMPT.txt")
    parser.add_argument("--output-dir", default="artifacts/model_rlplus")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--off-policy-ratio", type=float, default=0.5)
    parser.add_argument("--lambda-off", type=float, default=0.5)
    parser.add_argument("--off-min-clip", type=float, default=0.2)
    parser.add_argument("--off-max-clip", type=float, default=5.0)
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
    train_rows = Data.from_jsonl_file(args.train)
    gold_rows = read_jsonl(args.gold)

    dataset = RLPlusIterableDataset(
        iter_mixed(
            train_rows=train_rows,
            gold_rows=gold_rows,
            system_prompt=system_prompt,
            seed=args.seed,
            off_policy_ratio=args.off_policy_ratio,
        )
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.init_model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    running = {
        "reward_sum": 0.0,
        "entropy_proxy_sum": 0.0,
        "length_sum": 0,
        "count": 0,
        "off_count": 0,
    }

    def rlplus_reward_func(
        prompts: List[Any],
        completions: List[str],
        reference: List[Dict[str, Any]],
        off_policy: List[bool],
        teacher: List[str],
        **kwargs,
    ):
        rewards: List[float] = []
        for comp, ref, off, tch in zip(completions, reference, off_policy, teacher):
            data = Data.from_json_dict(ref)
            base_reward = 1.0 if env.verify(data, comp) else 0.0
            bonus = _exploration_bonus(comp)
            off_term = 0.0
            if off:
                match = _teacher_match_score(comp, tch)
                # clipped importance-style proxy
                ratio = max(args.off_min_clip, min(args.off_max_clip, 1.0 + match))
                off_term = args.lambda_off * ratio * match
                running["off_count"] += 1

            reward = base_reward + bonus + off_term
            rewards.append(float(reward))

            running["reward_sum"] += float(reward)
            running["entropy_proxy_sum"] += float(len(set(comp.split())) / max(1, len(comp.split())))
            running["length_sum"] += len(comp.split())
            running["count"] += 1
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
        reward_funcs=[rlplus_reward_func],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    c = max(1, running["count"])
    summary = {
        "init_model": args.init_model,
        "train": args.train,
        "gold": args.gold,
        "steps": args.steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "off_policy_ratio": args.off_policy_ratio,
        "lambda_off": args.lambda_off,
        "off_min_clip": args.off_min_clip,
        "off_max_clip": args.off_max_clip,
        "reward_mean": running["reward_sum"] / c,
        "entropy_mean": running["entropy_proxy_sum"] / c,
        "mean_len": running["length_sum"] / c,
        "off_policy_used": running["off_count"],
    }
    Path(args.output_dir, "rlplus_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved RL-PLUS model to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
