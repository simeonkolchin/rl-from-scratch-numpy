from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from base.data import Data
from envs.logic_circuit.env import LogicCircuitEnv


def load_jsonl(path: Path) -> List[Data]:
    return Data.from_jsonl_file(str(path))

def load_system_prompt(path: str = "training/SYSTEM_PROMPT.txt") -> str:
    return Path(path).read_text(encoding="utf-8").strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs finetuned models with vLLM.")
    parser.add_argument("--baseline", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--finetuned", default="artifacts/model_grpo")
    parser.add_argument("--datasets-dir", default="datasets")
    parser.add_argument("--difficulties", nargs="*", type=int, default=[1, 3, 5, 7, 10])
    parser.add_argument("--out", default="artifacts/metrics.json")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    try:
        from vllm import LLM, SamplingParams  # type: ignore
    except Exception as e:
        raise SystemExit(
            "vllm is not installed (or failed to import). Install deps from training/requirements-train.txt.\n"
            f"Import error: {e}"
        )

    system_prompt = load_system_prompt()
    env = LogicCircuitEnv()
    datasets_dir = Path(args.datasets_dir)

    def eval_model(model_name: str) -> Dict[str, float]:
        llm = LLM(model=model_name)
        params = SamplingParams(temperature=0.0, max_tokens=int(args.max_tokens))
        acc: Dict[str, float] = {}
        for d in args.difficulties:
            path = datasets_dir / f"test_d{d}.jsonl"
            items = load_jsonl(path)
            prompts = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": it.question},
                ]
                for it in items
            ]
            outputs = llm.generate(prompts, params)
            correct = 0
            for it, out in zip(items, outputs):
                text = out.outputs[0].text
                if env.verify(it, text):
                    correct += 1
            acc[str(d)] = correct / max(1, len(items))
            print(f"{model_name} difficulty={d}: {acc[str(d)]:.4f}")
        return acc

    metrics = {
        "baseline": args.baseline,
        "finetuned": args.finetuned,
        "accuracy": {
            "baseline": eval_model(args.baseline),
            "finetuned": eval_model(args.finetuned),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote metrics to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
