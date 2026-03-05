from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.data import Data
from envs.logic_circuit.env import LogicCircuitEnv


def gen_split(name: str, total: int, seed: int, difficulties: List[int]) -> List[Data]:
    env = LogicCircuitEnv()
    out: List[Data] = []
    per_d = max(1, total // len(difficulties))
    count = 0
    for d in difficulties:
        n = per_d if d != difficulties[-1] else (total - count)
        items = env.generate(
            num_of_questions=n,
            max_attempts=200,
            difficulty=d,
            seed=seed + d * 1000,
            id_prefix=name,
        )
        out.extend(items)
        count += len(items)
    return out[:total]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate train_full/val_full datasets.")
    parser.add_argument("--train-out", default="datasets/train_full.jsonl")
    parser.add_argument("--val-out", default="datasets/val_full.jsonl")
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    difficulties = list(range(1, 11))
    train_items = gen_split("train", args.train_size, args.seed, difficulties)
    val_items = gen_split("val", args.val_size, args.seed + 999, difficulties)

    Data.to_jsonl_file(args.train_out, train_items)
    Data.to_jsonl_file(args.val_out, val_items)

    print(f"Wrote train: {len(train_items)} -> {args.train_out}")
    print(f"Wrote val: {len(val_items)} -> {args.val_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
