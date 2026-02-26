from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from envs.logic_circuit.env import LogicCircuitEnv
from base.data import Data
from app.schemas.requests import GenerateTestDatasetsRequest
from app.schemas.responses import GenerateTestDatasetsResponse

router = APIRouter(prefix="/api/v1")


@router.post("/datasets/test", response_model=GenerateTestDatasetsResponse, summary="Сгенерировать фиксированные тест-датасеты")
async def generate_test_datasets(req: GenerateTestDatasetsRequest):
    out_dir = Path(req.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path("artifacts") / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    env = LogicCircuitEnv()
    written = []
    for d in req.difficulties:
        file_path = out_dir / f"test_d{int(d)}.jsonl"
        if file_path.exists() and not req.force:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"File already exists: {file_path}. Use force=true.",
            )

        items = env.generate(
            num_of_questions=int(req.n_per_difficulty),
            max_attempts=200,
            difficulty=int(d),
            seed=int(req.seed) + int(d) * 1000,
        )
        with open(file_path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(it.to_json_str())
                f.write("\n")
        written.append(str(file_path))

    return {"written_files": written}

