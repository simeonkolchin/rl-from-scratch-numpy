from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional

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
) -> Generator[Dict[str, Any], None, None]:
    """
    Infinite iterable dataset for GRPO training.
    Each item contains a chat-style prompt and a reference Data payload for verification.
    """
    rng = random.Random(seed)
    if not difficulties:
        difficulties = list(range(1, 11))

    while True:
        d = rng.choice(difficulties)
        # Give each sample its own seed to diversify.
        sample_seed = rng.randint(0, 2**31 - 1)
        data = env.generate(num_of_questions=1, max_attempts=max_attempts, difficulty=d, seed=sample_seed)[0]

        yield {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data.question},
            ],
            "reference": data.to_json(),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump a finite slice of the train iterable to JSONL (debug).")
    parser.add_argument("--out", default="datasets/train_sample.jsonl")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = LogicCircuitEnv()
    system_prompt = load_system_prompt()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    it = iter_train(env=env, system_prompt=system_prompt, seed=int(args.seed))
    with open(out_path, "w", encoding="utf-8") as f:
        for _ in range(int(args.n)):
            f.write(json.dumps(next(it), ensure_ascii=False))
            f.write("\n")

    print(f"Wrote {args.n} samples to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

