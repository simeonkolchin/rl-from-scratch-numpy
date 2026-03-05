from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.data import Data
from envs.logic_circuit.env import LogicCircuitEnv
from training.common import add_sampling_args, load_system_prompt, parse_k_list, pass_at_k, summarize_lengths


def load_generator(model_name: str):
    try:
        from vllm import LLM  # type: ignore

        llm = LLM(model=model_name)

        def run(prompts: List[List[Dict[str, str]]], temperature: float, top_p: float, max_tokens: int, n: int):
            from vllm import SamplingParams  # type: ignore

            params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n)
            outs = llm.generate(prompts, params)
            result: List[List[str]] = []
            for o in outs:
                result.append([x.text for x in o.outputs])
            return result

        return run, "vllm"
    except Exception:
        from transformers import pipeline  # type: ignore

        pipe = pipeline("text-generation", model=model_name)

        def run(prompts: List[List[Dict[str, str]]], temperature: float, top_p: float, max_tokens: int, n: int):
            result: List[List[str]] = []
            for p in prompts:
                text = "\n".join([f"{m['role']}: {m['content']}" for m in p])
                outs = pipe(
                    text,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    num_return_sequences=n,
                    return_full_text=False,
                )
                result.append([x["generated_text"] for x in outs])
            return result

        return run, "transformers"


def eval_dataset(
    model: str,
    dataset: str,
    n: int,
    k_list: List[int],
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Dict:
    env = LogicCircuitEnv()
    items = Data.from_jsonl_file(dataset)
    run, backend = load_generator(model)

    prompts = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": it.question}] for it in items]
    outputs = run(prompts, temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n)

    per_item_correct: List[int] = []
    lengths: List[int] = []
    proof_rows: List[Dict] = []

    for it, cands in zip(items, outputs):
        c = 0
        for cand in cands:
            lengths.append(len(cand.split()))
            ok = env.verify(it, cand)
            if ok:
                c += 1
        per_item_correct.append(c)
        proof_rows.append({"id": it.id, "n": n, "c": c})

    passk = {}
    for k in k_list:
        values = [pass_at_k(n=n, c=c, k=k) for c in per_item_correct]
        passk[str(k)] = float(sum(values) / max(1, len(values)))

    length_stats = summarize_lengths(lengths)
    return {
        "model": model,
        "backend": backend,
        "dataset": dataset,
        "n": n,
        "k_list": k_list,
        "pass_at_k": passk,
        "mean_len": length_stats["mean_len"],
        "median_len": length_stats["median_len"],
        "proof": proof_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate pass@k with n independent samples.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--system-prompt", default="training/SYSTEM_PROMPT.txt")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k-list", default="1,4,8,16,32,64,128")
    parser.add_argument("--out", default="artifacts/eval_passk.json")
    add_sampling_args(parser)
    args = parser.parse_args()

    k_list = parse_k_list(args.k_list)
    system_prompt = load_system_prompt(args.system_prompt)
    metrics = eval_dataset(
        model=args.model,
        dataset=args.dataset,
        n=args.n,
        k_list=k_list,
        system_prompt=system_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
