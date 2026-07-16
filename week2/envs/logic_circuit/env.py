from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence

from base.data import Data
from base.env import Env
from envs.logic_circuit.prompt import build_prompt
from envs.logic_circuit.verifier import LogicCircuitVerifier


_ALL_GATES: List[str] = ["AND", "OR", "NOT", "XOR", "NAND", "NOR", "XNOR"]


def difficulty_to_hyperparams(difficulty: int) -> Dict[str, Any]:
    d = int(difficulty)
    if d < 1 or d > 10:
        raise ValueError("difficulty must be in [1, 10]")

    num_inputs = 2 + (d - 1) // 2
    num_inputs = max(2, min(6, num_inputs))

    num_gates = 2 + 2 * (d - 1)
    num_gates = max(2, min(20, num_gates))

    if 1 <= d <= 3:
        allowed = ["AND", "OR", "NOT"]
    elif 4 <= d <= 6:
        allowed = ["AND", "OR", "NOT", "XOR", "NAND"]
    elif 7 <= d <= 8:
        allowed = ["AND", "OR", "NOT", "XOR", "NAND", "NOR"]
    else:
        allowed = ["AND", "OR", "NOT", "XOR", "NAND", "NOR", "XNOR"]

    return {
        "num_inputs": num_inputs,
        "num_gates": num_gates,
        "allowed_gates": allowed,
        "require_non_constant": True,
    }


class LogicCircuitEnv(Env):
    def __init__(self):
        super().__init__(name="logic_circuit", verifier=LogicCircuitVerifier)

    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        difficulty: Optional[int] = 1,
        **kwargs,
    ) -> List[Data]:
        seed = kwargs.pop("seed", None)
        base_rng = random.Random(seed)

        if difficulty is None:
            difficulty = 1
        difficulty = int(difficulty)

        defaults = difficulty_to_hyperparams(difficulty)
        config = {**defaults, **kwargs}
        config = self._normalize_config(config)

        items: List[Data] = []
        for i in range(int(num_of_questions)):
            rng = random.Random(base_rng.randint(0, 2**63 - 1))
            gates, inputs = self._sample_instance(rng=rng, config=config, max_attempts=max_attempts)
            y = self.verifier.simulate(inputs, gates)

            metadata = {
                "env": self.name,
                "config": config,
                "inputs": inputs,
                "gates": gates,
            }
            question = build_prompt(config=config, inputs=inputs, gates=gates)
            items.append(Data(question=question, answer=str(y), difficulty=difficulty, metadata=metadata))

        return items

    def extract_answer(self, test_solution: str) -> Optional[str]:
        return self.verifier.extract_answer(test_solution)

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        num_inputs = int(config.get("num_inputs", 2))
        num_gates = int(config.get("num_gates", 2))
        require_non_constant = bool(config.get("require_non_constant", True))

        allowed = config.get("allowed_gates") or ["AND", "OR", "NOT"]
        allowed = [str(x).upper() for x in allowed]
        for op in allowed:
            if op not in _ALL_GATES:
                raise ValueError(f"Unsupported gate in allowed_gates: {op}")

        num_inputs = max(2, min(6, num_inputs))
        num_gates = max(2, min(20, num_gates))

        return {
            "num_inputs": num_inputs,
            "num_gates": num_gates,
            "allowed_gates": allowed,
            "require_non_constant": require_non_constant,
        }

    def _sample_instance(self, rng: random.Random, config: Dict[str, Any], max_attempts: int) -> tuple[List[Dict[str, Any]], List[int]]:
        num_inputs = config["num_inputs"]
        num_gates = config["num_gates"]
        allowed = config["allowed_gates"]
        require_non_constant = config["require_non_constant"]

        for _ in range(int(max_attempts)):
            gates = self._sample_gates(rng=rng, num_inputs=num_inputs, num_gates=num_gates, allowed_gates=allowed)
            if require_non_constant and not self.verifier.is_non_constant(num_inputs=num_inputs, gates=gates):
                continue
            inputs = [rng.randint(0, 1) for _ in range(num_inputs)]
            return gates, inputs

        raise RuntimeError("Failed to generate a valid circuit within max_attempts")

    def _sample_gates(
        self,
        rng: random.Random,
        num_inputs: int,
        num_gates: int,
        allowed_gates: Sequence[str],
    ) -> List[Dict[str, Any]]:
        sources: List[str] = [f"x{i}" for i in range(num_inputs)]
        gates: List[Dict[str, Any]] = []
        for idx in range(num_gates):
            op = rng.choice(list(allowed_gates))
            if op == "NOT":
                a = rng.choice(sources)
                gates.append({"op": "NOT", "a": a})
            else:
                a = rng.choice(sources)
                b = rng.choice(sources)
                gates.append({"op": op, "a": a, "b": b})
            sources.append(f"g{idx}")
        return gates
