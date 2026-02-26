from envs.logic_circuit.env import LogicCircuitEnv


def test_generate_respects_hyperparams():
    env = LogicCircuitEnv()
    items = env.generate(
        num_of_questions=3,
        difficulty=1,
        num_inputs=6,
        num_gates=5,
        allowed_gates=["AND", "OR", "NOT"],
        seed=123,
    )
    assert len(items) == 3
    for it in items:
        cfg = it.metadata["config"]
        assert cfg["num_inputs"] == 6
        assert cfg["num_gates"] == 5
        assert set(cfg["allowed_gates"]) == {"AND", "OR", "NOT"}


def test_generate_non_constant_default():
    env = LogicCircuitEnv()
    items = env.generate(num_of_questions=5, difficulty=5, seed=999)
    for it in items:
        cfg = it.metadata["config"]
        assert cfg["require_non_constant"] is True
        assert env.verifier.is_non_constant(cfg["num_inputs"], it.metadata["gates"]) is True

