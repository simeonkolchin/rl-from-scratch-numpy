from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Type

from base.data import Data
from base.verifier import Verifier


class Env(ABC):
    def __init__(self, name: str, verifier: Type[Verifier]):
        self.name = name
        self.verifier = verifier()

    @abstractmethod
    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        difficulty: Optional[int] = 1,
        **kwargs,
    ) -> List[Data]:
        raise NotImplementedError

    def verify(self, data: Data, test_solution: str) -> bool:
        return self.verifier.verify(data, test_solution)

    def extract_answer(self, test_solution: str):
        return self.verifier.extract_answer(test_solution)
