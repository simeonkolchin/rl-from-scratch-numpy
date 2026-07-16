from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    num_of_questions: int = Field(default=100, ge=1)
    max_attempts: int = Field(default=100, ge=1)
    difficulty: Optional[int] = Field(default=1, ge=1, le=10)
    hyperparams: Dict[str, Any] = Field(default_factory=dict)
    seed: Optional[int] = None


class VerifyRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Data JSON payload")
    test_solution: str


class GenerateTestDatasetsRequest(BaseModel):
    out_dir: str = Field(default="datasets", description="Relative to /app/artifacts unless absolute.")
    difficulties: List[int] = Field(default_factory=lambda: [1, 3, 5, 7, 10])
    n_per_difficulty: int = Field(default=200, ge=1)
    seed: int = Field(default=1234)
    force: bool = Field(default=False)

