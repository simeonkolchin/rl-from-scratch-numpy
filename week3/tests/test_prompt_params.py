from training.common import load_system_prompt


def test_system_prompt_exists():
    p = load_system_prompt("training/SYSTEM_PROMPT.txt")
    assert "<think>" in p and "<answer>" in p
