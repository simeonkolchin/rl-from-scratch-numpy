from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GenerateResponse(BaseModel):
    items: List[Dict[str, Any]]


class VerifyResponse(BaseModel):
    correct: bool
    expected: str
    extracted: Optional[str] = None
    error: Optional[str] = None


class EnvInfo(BaseModel):
    name: str
    description: str
    hyperparams: Dict[str, str]


class ListEnvsResponse(BaseModel):
    envs: List[EnvInfo]


class GenerateTestDatasetsResponse(BaseModel):
    written_files: List[str] = Field(default_factory=list)

