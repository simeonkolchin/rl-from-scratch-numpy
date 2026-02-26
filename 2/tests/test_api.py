import os
import sys
from pathlib import Path

import pytest


def test_api_generate_verify_roundtrip():
    pytest.importorskip("fastapi")
    pytest.importorskip("starlette")

    os.environ["RL_ENVS_SAVE_OPENAPI_SPEC"] = "0"

    service_dir = Path(__file__).resolve().parents[1] / "rl-envs-service"
    sys.path.insert(0, str(service_dir))

    from fastapi.testclient import TestClient  # type: ignore

    from app.app import app  # type: ignore

    client = TestClient(app)

    r = client.post(
        "/api/v1/envs/logic_circuit/generate",
        json={"num_of_questions": 1, "difficulty": 2, "seed": 1},
    )
    assert r.status_code == 200
    payload = r.json()
    item = payload["items"][0]

    r2 = client.post(
        "/api/v1/envs/logic_circuit/verify",
        json={"data": item, "test_solution": f"<answer>{item['answer']}</answer>"},
    )
    assert r2.status_code == 200
    assert r2.json()["correct"] is True

