from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.data import Data
from training.common import load_system_prompt
from training.eval_passk import eval_dataset


def select_hard(
    model: str,
    items: List[Data],
    system_prompt: str,
    n_screen: int,
    n_final: int,
    hard_size: int,
    sampling: Dict,
    tmp_path: Path,
) -> List[Data]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tmp_dataset = tmp_path / "_subset_tmp.jsonl"
    Data.to_jsonl_file(tmp_dataset, items)

    m1 = eval_dataset(
        model=model,
        dataset=str(tmp_dataset),
        n=n_screen,
        k_list=[n_screen],
        system_prompt=system_prompt,
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        max_tokens=sampling["max_tokens"],
    )
    screen_hard_ids = {row["id"] for row in m1["proof"] if row["c"] == 0}
    screened = [x for x in items if x.id in screen_hard_ids]

    Data.to_jsonl_file(tmp_dataset, screened)
    m2 = eval_dataset(
        model=model,
        dataset=str(tmp_dataset),
        n=n_final,
        k_list=[n_final],
        system_prompt=system_prompt,
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        max_tokens=sampling["max_tokens"],
    )

    final_hard_ids = {row["id"] for row in m2["proof"] if row["c"] == 0}
    hard = [x for x in screened if x.id in final_hard_ids][:hard_size]

    proof = {
        "screen": {
            "n": n_screen,
            "total": len(items),
            "hard_candidates": len(screened),
            "pass_at_n": m1["pass_at_k"].get(str(n_screen), None),
        },
        "final": {
            "n": n_final,
            "c_eq_0": len(final_hard_ids),
            "selected": len(hard),
            "pass_at_n": m2["pass_at_k"].get(str(n_final), None),
        },
        "rows": m2["proof"],
    }
    (tmp_path / "proof.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
    return hard


def main() -> int:
    parser = argparse.ArgumentParser(description="Build hard subsets with pass@128=0 criterion.")
    parser.add_argument("--train", default="datasets/train_full.jsonl")
    parser.add_argument("--val", default="datasets/val_full.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--system-prompt", default="training/SYSTEM_PROMPT.txt")
    parser.add_argument("--n-screen", type=int, default=16)
    parser.add_argument("--n-final", type=int, default=128)
    parser.add_argument("--hard-train-size", type=int, default=512)
    parser.add_argument("--hard-val-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--hard-train-out", default="datasets/hard_train.jsonl")
    parser.add_argument("--hard-val-out", default="datasets/hard_val.jsonl")
    parser.add_argument("--proof-out", default="artifacts/hard_subset_proof.json")
    args = parser.parse_args()

    sampling = {"temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens}
    system_prompt = load_system_prompt(args.system_prompt)

    train_items = Data.from_jsonl_file(args.train)
    val_items = Data.from_jsonl_file(args.val)

    tmp = Path("artifacts/hard_subset_tmp")
    tmp.mkdir(parents=True, exist_ok=True)

    hard_train = select_hard(
        model=args.model,
        items=train_items,
        system_prompt=system_prompt,
        n_screen=args.n_screen,
        n_final=args.n_final,
        hard_size=args.hard_train_size,
        sampling=sampling,
        tmp_path=tmp / "train",
    )
    hard_val = select_hard(
        model=args.model,
        items=val_items,
        system_prompt=system_prompt,
        n_screen=args.n_screen,
        n_final=args.n_final,
        hard_size=args.hard_val_size,
        sampling=sampling,
        tmp_path=tmp / "val",
    )

    Data.to_jsonl_file(args.hard_train_out, hard_train)
    Data.to_jsonl_file(args.hard_val_out, hard_val)

    proof = {
        "train_total": len(train_items),
        "val_total": len(val_items),
        "hard_train": len(hard_train),
        "hard_val": len(hard_val),
        "criterion": f"pass@{args.n_final}=0",
    }
    Path(args.proof_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.proof_out).write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"hard_train={len(hard_train)} -> {args.hard_train_out}")
    print(f"hard_val={len(hard_val)} -> {args.hard_val_out}")
    print(f"proof -> {args.proof_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
