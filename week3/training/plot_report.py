from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # type: ignore


def load(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plot_passk(curves: Dict[str, Dict[str, float]], out: Path, title: str) -> None:
    plt.figure(figsize=(8, 5))
    for name, kv in curves.items():
        ks = sorted(int(k) for k in kv.keys())
        ys = [kv[str(k)] for k in ks]
        plt.plot(ks, ys, marker="o", label=name)
    plt.xlabel("k")
    plt.ylabel("pass@k")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build report plots from eval JSONs.")
    parser.add_argument("--baseline-full", required=True)
    parser.add_argument("--baseline-hard", required=True)
    parser.add_argument("--grpo-full", required=True)
    parser.add_argument("--grpo-hard", required=True)
    parser.add_argument("--sftgrpo-full", required=True)
    parser.add_argument("--sftgrpo-hard", required=True)
    parser.add_argument("--rlplus-full", required=True)
    parser.add_argument("--rlplus-hard", required=True)
    parser.add_argument("--out-dir", default="reports/figures")
    args = parser.parse_args()

    bf = load(args.baseline_full)
    bh = load(args.baseline_hard)
    gf = load(args.grpo_full)
    gh = load(args.grpo_hard)
    sf = load(args.sftgrpo_full)
    sh = load(args.sftgrpo_hard)
    rf = load(args.rlplus_full)
    rh = load(args.rlplus_hard)

    out_dir = Path(args.out_dir)
    plot_passk(
        {
            "baseline": bf["pass_at_k"],
            "grpo": gf["pass_at_k"],
            "sft_grpo": sf["pass_at_k"],
            "rlplus": rf["pass_at_k"],
        },
        out_dir / "passk_full.png",
        "pass@k on val(full)",
    )
    plot_passk(
        {
            "baseline": bh["pass_at_k"],
            "grpo": gh["pass_at_k"],
            "sft_grpo": sh["pass_at_k"],
            "rlplus": rh["pass_at_k"],
        },
        out_dir / "passk_hard.png",
        "pass@k on hard_val",
    )

    print(f"Wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
