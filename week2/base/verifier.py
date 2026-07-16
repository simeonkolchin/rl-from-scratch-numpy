from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from base.data import Data


class Verifier(ABC):
    """Base class for verifier."""

    def __init__(self):
        pass

    @abstractmethod
    def verify(self, data: Data, test_solution: str) -> bool:
        """
        Verify whether the test solution is consistent with the gold answer.
        @param data: Data
        @param test_solution: str
        @return: bool
        """

    @abstractmethod
    def extract_answer(self, test_solution: str) -> Optional[str]:
        """
        Extract the answer from the test solution.
        @param test_solution: str
        @return: Optional[str]
        """

