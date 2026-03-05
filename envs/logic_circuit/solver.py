from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .verifier import LogicCircuitVerifier


def solve_with_trace(inputs: List[int], gates: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
    verifier = LogicCircuitVerifier()
    values: Dict[str, int] = {f"x{i}": int(v) for i, v in enumerate(inputs)}
    trace: List[str] = ["Evaluate gates in order:"]
    for idx, g in enumerate(gates):
        op = g["op"]
        a_name = g["a"]
        a = values[a_name]
        if op == "NOT":
            out = 1 - a
            trace.append(f"g{idx} = NOT({a_name}={a}) = {out}")
        else:
            b_name = g["b"]
            b = values[b_name]
            out = verifier._apply_binary(op, a, b)
            trace.append(f"g{idx} = {op}({a_name}={a}, {b_name}={b}) = {out}")
        values[f"g{idx}"] = int(out)

    y = int(values[f"g{len(gates)-1}"])
    trace.append(f"Final output is g{len(gates)-1} = {y}")
    return y, trace


def build_gold_completion(answer: int, trace: List[str]) -> str:
    think = "\n".join(trace)
    return f"<think>\n{think}\n</think>\n<answer>{int(answer)}</answer>"
