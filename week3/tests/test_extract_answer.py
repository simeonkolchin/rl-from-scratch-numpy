from envs.logic_circuit.verifier import LogicCircuitVerifier


def test_extract_answer_tag():
    v = LogicCircuitVerifier()
    assert v.extract_answer("<answer>1</answer>") == "1"


def test_extract_answer_json():
    v = LogicCircuitVerifier()
    assert v.extract_answer('{"answer": 0}') == "0"


def test_extract_answer_last_bit():
    v = LogicCircuitVerifier()
    assert v.extract_answer("foo\n1\n") == "1"
