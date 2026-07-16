from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from base.data import Data
from envs.logic_circuit.env import LogicCircuitEnv
from app.schemas.requests import GenerateRequest, VerifyRequest
from app.schemas.responses import GenerateResponse, ListEnvsResponse, VerifyResponse

router = APIRouter(prefix="/api/v1")


_ENVS = {
    "logic_circuit": LogicCircuitEnv(),
}


@router.get("/envs", response_model=ListEnvsResponse, summary="Список доступных сред")
async def list_envs():
    return {
        "envs": [
            {
                "name": "logic_circuit",
                "description": "Evaluate a random Boolean logic circuit output bit.",
                "hyperparams": {
                    "num_inputs": "int (2..6)",
                    "num_gates": "int (2..20)",
                    "allowed_gates": "list[str]",
                    "require_non_constant": "bool",
                    "seed": "int",
                },
            }
        ]
    }


@router.post("/envs/{env_name}/generate", response_model=GenerateResponse, summary="Сгенерировать задачи")
async def generate(env_name: str, req: GenerateRequest):
    env = _ENVS.get(env_name)
    if env is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown env")

    hyper = dict(req.hyperparams or {})
    if req.seed is not None:
        hyper["seed"] = int(req.seed)

    items = env.generate(
        num_of_questions=int(req.num_of_questions),
        max_attempts=int(req.max_attempts),
        difficulty=req.difficulty,
        **hyper,
    )
    return {"items": [it.to_json() for it in items]}


@router.post("/envs/{env_name}/verify", response_model=VerifyResponse, summary="Проверить решение")
async def verify(env_name: str, req: VerifyRequest):
    env = _ENVS.get(env_name)
    if env is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown env")

    try:
        data = Data.from_json_dict(req.data)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid data payload: {e}")

    extracted = env.extract_answer(req.test_solution)
    correct = env.verify(data, req.test_solution)
    return {
        "correct": bool(correct),
        "expected": (data.answer or "").strip(),
        "extracted": extracted,
        "error": None if extracted is not None else "failed_to_extract_answer",
    }
