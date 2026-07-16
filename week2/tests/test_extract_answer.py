import pytest

from envs.logic_circuit.verifier import LogicCircuitVerifier


@pytest.mark.parametrize(
    "text,expected",
    [
        ("<answer>0</answer>", "0"),
        ("<answer>\n 1 \n</answer>", "1"),
        ("{\"answer\": 0}", "0"),
        ("some text\n0\n", "0"),
        ("no answer here", None),
    ],
)
def test_extract_answer(text, expected):
    v = LogicCircuitVerifier()
    assert v.extract_answer(text) == expected

