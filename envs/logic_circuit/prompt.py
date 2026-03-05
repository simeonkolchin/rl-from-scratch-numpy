from __future__ import annotations

from typing import Any, Dict, List


def build_prompt(config: Dict[str, Any], inputs: List[int], gates: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("You are given a boolean logic circuit.")
    lines.append("Compute the final output bit of the last gate.")
    lines.append("Use <think>...</think> for reasoning and put final bit in <answer>0</answer> or <answer>1</answer>.")
    lines.append("")
    lines.append(f"Config: num_inputs={config['num_inputs']}, num_gates={config['num_gates']}")
    lines.append("Inputs:")
    for i, bit in enumerate(inputs):
        lines.append(f"  x{i} = {bit}")
    lines.append("Gates:")
    for i, g in enumerate(gates):
        op = g["op"]
        if op == "NOT":
            lines.append(f"  g{i} = NOT({g['a']})")
        else:
            lines.append(f"  g{i} = {op}({g['a']}, {g['b']})")
    lines.append("Return only one final bit in <answer> tags.")
    return "\n".join(lines)
