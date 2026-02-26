from __future__ import annotations

from typing import Any, Dict, List


def build_prompt(config: Dict[str, Any], inputs: List[int], gates: List[Dict[str, Any]]) -> str:
    """
    Build an unambiguous English prompt for a concrete logic-circuit instance.
    inputs: list of 0/1 values for x0..x{n-1}
    gates: list of gate dicts in topological order, g0..g{m-1}
    """
    n = config["num_inputs"]
    m = config["num_gates"]
    allowed_gates = ", ".join(config["allowed_gates"])

    inputs_block = "\n".join(f"- x{i} = {v}" for i, v in enumerate(inputs))

    gate_lines: List[str] = []
    for idx, g in enumerate(gates):
        op = g["op"]
        if op == "NOT":
            gate_lines.append(f"- g{idx} = NOT({g['a']})")
        else:
            gate_lines.append(f"- g{idx} = {op}({g['a']}, {g['b']})")
    gates_block = "\n".join(gate_lines)

    prompt = f"""
    You are given a Boolean logic circuit. All values are bits: 0 or 1.

    Gate definitions:
    - NOT(a) = 1 if a=0 else 0
    - AND(a,b) = 1 iff a=1 and b=1
    - OR(a,b)  = 1 iff at least one of a,b is 1
    - XOR(a,b) = 1 iff a and b are different
    - XNOR(a,b)= 1 iff a and b are equal
    - NAND(a,b)= NOT(AND(a,b))
    - NOR(a,b) = NOT(OR(a,b))

    Task:
    - There are {n} inputs: x0..x{n-1}.
    - There are {m} gates: g0..g{m-1}, evaluated in order.
    - Allowed gate types for this instance: {allowed_gates}.
    - The circuit output y is the value of the last gate: y = g{m-1}.

    Instance:
    Inputs:
    {inputs_block}

    Gates:
    {gates_block}

    Your answer must be in the following format:
    <answer>y</answer>
    where y is 0 or 1. Output nothing else.
    """

    return prompt

