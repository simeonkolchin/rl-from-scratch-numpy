from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List

from base.data import Data


def load_system_prompt(path: str = "training/SYSTEM_PROMPT.txt") -> str:
    return Path(path).read_text(encoding="utf-8").strip() + "\n"


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def load_data_jsonl(path: str | Path) -> List[Data]:
    return Data.from_jsonl_file(path)


def save_data_jsonl(path: str | Path, items: Iterable[Data]) -> None:
    Data.to_jsonl_file(path, items)


def pass_at_k(n: int, c: int, k: int) -> float:
    if c <= 0:
        return 0.0
    if k > n:
        k = n
    if n - c < k:
        return 1.0
    return 1.0 - (comb(n - c, k) / comb(n, k))


def summarize_lengths(lengths: List[int]) -> Dict[str, float]:
    if not lengths:
        return {"mean_len": 0.0, "median_len": 0.0}
    return {
        "mean_len": float(mean(lengths)),
        "median_len": float(median(lengths)),
    }


def parse_k_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def add_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=256)
