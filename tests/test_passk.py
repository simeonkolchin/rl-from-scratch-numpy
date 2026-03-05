from training.common import pass_at_k


def test_passk_zero():
    assert pass_at_k(n=128, c=0, k=128) == 0.0


def test_passk_all_correct():
    assert pass_at_k(n=16, c=16, k=8) == 1.0


def test_passk_monotonicity():
    p1 = pass_at_k(n=32, c=1, k=1)
    p8 = pass_at_k(n=32, c=1, k=8)
    assert p8 >= p1
