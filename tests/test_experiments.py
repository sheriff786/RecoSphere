"""Phase 8: assignment properties."""
import numpy as np
import pytest

from serve.experiments import Experiment, bucket


def exp(salt="s1", traffic=(0.5, 0.5)):
    return Experiment({"name": "e", "salt": salt, "arms": [
        {"name": "control", "model": "production", "traffic": traffic[0]},
        {"name": "treatment", "model": "runs/x", "traffic": traffic[1]}]})


def test_deterministic_and_balanced():
    e = exp()
    users = [f"U{i}" for i in range(40_000)]
    a = [e.assign(u)["name"] for u in users]
    assert a == [e.assign(u)["name"] for u in users]                     # same user -> same arm, always
    share = np.mean([x == "treatment" for x in a])
    assert abs(share - 0.5) < 0.01


def test_uneven_split():
    e = exp(traffic=(0.9, 0.1))
    share = np.mean([e.assign(f"U{i}")["name"] == "treatment" for i in range(40_000)])
    assert abs(share - 0.1) < 0.006


def test_new_salt_reshuffles_users():
    a = [exp("s1").assign(f"U{i}")["name"] for i in range(20_000)]
    b = [exp("s2").assign(f"U{i}")["name"] for i in range(20_000)]
    agree = np.mean([x == y for x, y in zip(a, b)])
    assert abs(agree - 0.5) < 0.02                                       # independent of the old experiment


def test_bucket_range_and_bad_configs():
    assert all(0 <= bucket("s", f"u{i}") < 10_000 for i in range(1000))
    with pytest.raises(AssertionError):
        exp(traffic=(0.6, 0.6))
    with pytest.raises(AssertionError):
        Experiment({"name": "e", "salt": "s", "arms": [{"name": "a", "model": "production", "traffic": .5},
                                                       {"name": "b", "model": "runs/x", "traffic": .5}]})
