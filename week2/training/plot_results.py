from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot paired bar chart from artifacts/metrics.json.")
    parser.add_argument("--metrics", default="artifacts/metrics.json")
    parser.add_argument("--out", default="artifacts/accuracy_bars.png")
    args = parser.parse_args()

    import matplotlib.pyplot as plt  # type: ignore

    metrics_path = Path(args.metrics)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    base = payload["accuracy"]["baseline"]
    fine = payload["accuracy"]["finetuned"]

    diffs: List[str] = sorted(base.keys(), key=lambda x: int(x))
    base_vals = [base[d] for d in diffs]
    fine_vals = [fine[d] for d in diffs]

    x = list(range(len(diffs)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([i - width / 2 for i in x], base_vals, width, label="baseline")
    ax.bar([i + width / 2 for i in x], fine_vals, width, label="finetuned")

    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Difficulty")
    ax.set_xticks(x)
    ax.set_xticklabels(diffs)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Baseline vs Finetuned accuracy by difficulty")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved plot to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

