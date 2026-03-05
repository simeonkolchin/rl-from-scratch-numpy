from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Data:
    id: str
    question: str
    answer: str
    difficulty: int
    metadata: Dict[str, Any]

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
        }

    def to_json_str(self) -> str:
        return json.dumps(self.to_json(), ensure_ascii=False)

    @staticmethod
    def from_json_dict(payload: Dict[str, Any]) -> "Data":
        return Data(
            id=str(payload.get("id", "")),
            question=str(payload["question"]),
            answer=str(payload["answer"]),
            difficulty=int(payload.get("difficulty", 1)),
            metadata=dict(payload.get("metadata", {})),
        )

    @staticmethod
    def from_jsonl_file(path: str | Path) -> List["Data"]:
        items: List[Data] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                items.append(Data.from_json_dict(json.loads(s)))
        return items

    @staticmethod
    def to_jsonl_file(path: str | Path, items: Iterable["Data"]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for it in items:
                f.write(it.to_json_str())
                f.write("\n")
