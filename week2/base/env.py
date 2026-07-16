from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Type

from base.data import Data
from base.verifier import Verifier


class Env(ABC):
    """
    Base class for game/environment.
    @param name: name of the game
    @param verifier: class of the verifier
    """

    def __init__(self, name: str, verifier: Type[Verifier]):
        self.name = name
        self.verifier = verifier()

    @abstractmethod
    def generate(self, num_of_questions: int = 100, max_attempts: int = 100, difficulty: Optional[int] = 1, **kwargs) -> List[Data]:
        """
        Generate game questions and answers.
        Supports both difficulty and direct hyperparameters via kwargs.
        @return: list of Data
        """
        raise NotImplementedError("Env.generate() is not implemented")

    def verify(self, data: Data, test_solution: str) -> bool:
        """
        Verify whether the test solution is consistent with the gold answer.
        @return: bool
        """
        return self.verifier.verify(data, test_solution)

    @abstractmethod
    def extract_answer(self, test_solution: str) -> Optional[str]:
        """Extract the answer from the test solution."""
        raise NotImplementedError("Env.extract_answer() is not implemented")

