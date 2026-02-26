import pytest

from envs.logic_circuit.verifier import LogicCircuitVerifier


def test_simulation_correctness_small():
    v = LogicCircuitVerifier()
    # g0 = AND(x0, x1)
    # g1 = NOT(g0)
    gates = [
        {"op": "AND", "a": "x0", "b": "x1"},
        {"op": "NOT", "a": "g0"},
    ]
    assert v.simulate([0, 0], gates) == 1
    assert v.simulate([0, 1], gates) == 1
    assert v.simulate([1, 0], gates) == 1
    assert v.simulate([1, 1], gates) == 0


def test_is_non_constant():
    v = LogicCircuitVerifier()
    gates = [{"op": "XOR", "a": "x0", "b": "x1"}]
    assert v.is_non_constant(num_inputs=2, gates=gates) is True

