from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from base.data import Data
from base.verifier import Verifier


_ANSWER_TAG_RE = re.compile(r"<answer>\s*([01])\s*</answer>", re.IGNORECASE | re.DOTALL)
_LAST_BIT_RE = re.compile(r"(^|\n)\s*([01])\s*($|\n)")


class LogicCircuitVerifier(Verifier):
    def extract_answer(self, test_solution: str) -> Optional[str]:
        if test_solution is None:
            return None

        m = _ANSWER_TAG_RE.search(test_solution)
        if m:
            return m.group(1)

        s = test_solution.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                payload = json.loads(s)
                if isinstance(payload, dict) and "answer" in payload:
                    v = payload["answer"]
                    if v in (0, 1, "0", "1"):
                        return str(v)
            except Exception:
                pass

        matches = list(_LAST_BIT_RE.finditer(test_solution))
        if matches:
            return matches[-1].group(2)
        return None

    def verify(self, data: Data, test_solution: str) -> bool:
        extracted = self.extract_answer(test_solution)
        if extracted is None:
            return False
        expected = (data.answer or "").strip()
        return extracted.strip() == expected

    def simulate(self, inputs: List[int], gates: List[Dict[str, Any]]) -> int:
        values: Dict[str, int] = {}
        for i, v in enumerate(inputs):
            values[f"x{i}"] = int(v)

        for idx, g in enumerate(gates):
            op = g["op"]
            a = self._get(values, g.get("a"))
            if op == "NOT":
                out = 1 - a
            else:
                b = self._get(values, g.get("b"))
                out = self._apply_binary(op, a, b)
            values[f"g{idx}"] = out

        if not gates:
            raise ValueError("Circuit must contain at least one gate")
        return values[f"g{len(gates) - 1}"]

    def is_non_constant(self, num_inputs: int, gates: List[Dict[str, Any]]) -> bool:
        seen: set[int] = set()
        for mask in range(1 << num_inputs):
            inputs = [(mask >> i) & 1 for i in range(num_inputs)]
            y = self.simulate(inputs, gates)
            seen.add(y)
            if len(seen) > 1:
                return True
        return False

    def _get(self, values: Dict[str, int], src: Optional[str]) -> int:
        if not src or src not in values:
            raise ValueError(f"Invalid source reference: {src!r}")
        v = values[src]
        if v not in (0, 1):
            raise ValueError(f"Non-bit value for {src}: {v!r}")
        return int(v)

    def _apply_binary(self, op: str, a: int, b: int) -> int:
        if op == "AND":
            return int(a & b)
        if op == "OR":
            return int(a | b)
        if op == "XOR":
            return int(a ^ b)
        if op == "XNOR":
            return int(1 - (a ^ b))
        if op == "NAND":
            return int(1 - (a & b))
        if op == "NOR":
            return int(1 - (a | b))
        raise ValueError(f"Unsupported op: {op}")
