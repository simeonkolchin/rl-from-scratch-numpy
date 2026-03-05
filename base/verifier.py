from __future__ import annotations

from abc import ABC, abstractmethod
from base.data import Data


class Verifier(ABC):
    @abstractmethod
    def extract_answer(self, test_solution: str):
        raise NotImplementedError

    @abstractmethod
    def verify(self, data: Data, test_solution: str) -> bool:
        raise NotImplementedError
