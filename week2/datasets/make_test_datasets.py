from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

from base.data import Data
from envs.logic_circuit.env import LogicCircuitEnv


DEFAULT_DIFFICULTIES = [1, 3, 5, 7, 10]


def write_jsonl(path: Path, items: Iterable[Data]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.to_json_str())
            f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate fixed test datasets (do not resample).")
    parser.add_argument("--out-dir", default="datasets", help="Output directory for test_*.jsonl files.")
    parser.add_argument("--difficulties", nargs="*", type=int, default=DEFAULT_DIFFICULTIES)
    parser.add_argument("--n-per-difficulty", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()

    env = LogicCircuitEnv()
    out_dir = Path(args.out_dir)

    for d in args.difficulties:
        file_path = out_dir / f"test_d{d}.jsonl"
        if file_path.exists() and not args.force:
            raise SystemExit(f"Refusing to overwrite existing file: {file_path}. Use --force.")

        # Stable per-difficulty seed.
        items = env.generate(
            num_of_questions=int(args.n_per_difficulty),
            max_attempts=200,
            difficulty=int(d),
            seed=int(args.seed) + int(d) * 1000,
        )
        write_jsonl(file_path, items)
        print(f"Wrote {len(items)} items to {file_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

