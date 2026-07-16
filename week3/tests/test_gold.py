from base.data import Data
from envs.logic_circuit.env import LogicCircuitEnv
from envs.logic_circuit.solver import build_gold_completion, solve_with_trace


def test_gold_verified_true():
    env = LogicCircuitEnv()
    data = env.generate(num_of_questions=1, difficulty=4, seed=123, id_prefix="t")[0]
    ans, trace = solve_with_trace(data.metadata["inputs"], data.metadata["gates"])
    comp = build_gold_completion(ans, trace)
    assert env.verify(data, comp)
