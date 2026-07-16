from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.data import Data
from envs.logic_circuit.env import LogicCircuitEnv
from envs.logic_circuit.solver import build_gold_completion, solve_with_trace
from training.common import write_jsonl


def to_gold_row(data: Data, completion: str, verified: bool) -> Dict:
    return {
        "id": data.id,
        "prompt": data.question,
        "gold_completion": completion,
        "verified": bool(verified),
        "answer": data.answer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build algorithmic gold trajectories and verify each.")
    parser.add_argument("--train", default="datasets/train_full.jsonl")
    parser.add_argument("--out", default="datasets/gold_train.jsonl")
    parser.add_argument("--limit", type=int, default=4000)
    args = parser.parse_args()

    env = LogicCircuitEnv()
    items = Data.from_jsonl_file(args.train)[: args.limit]
    rows: List[Dict] = []
    bad = 0

    for it in items:
        inputs = it.metadata["inputs"]
        gates = it.metadata["gates"]
        answer, trace = solve_with_trace(inputs=inputs, gates=gates)
        completion = build_gold_completion(answer=answer, trace=trace)
        verified = env.verify(it, completion)
        if not verified:
            bad += 1
            continue
        rows.append(to_gold_row(it, completion, verified))

    write_jsonl(args.out, rows)
    print(f"Gold rows: {len(rows)}, dropped: {bad}, out: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
