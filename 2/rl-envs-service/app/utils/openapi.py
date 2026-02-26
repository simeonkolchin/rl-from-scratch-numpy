from __future__ import annotations

import os
from pathlib import Path

import yaml
from fastapi import FastAPI


def save_openapi_spec(app: FastAPI, path: str | None = None) -> None:
    if os.getenv("RL_ENVS_SAVE_OPENAPI_SPEC", "1") != "1":
        return

    base_dir = Path(__file__).resolve().parents[2]  # rl-envs-service/
    if path is None:
        path = str(base_dir / "openapi_spec" / "rl-envs-service-1.0.0.yaml")

    openapi_spec = app.openapi()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.dump(openapi_spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
