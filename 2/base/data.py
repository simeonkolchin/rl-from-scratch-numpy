from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Data:
    """
    Data class for game/corpus
    @param question: question of the game/corpus
    @param answer: answer of the game/corpus
    @param difficulty: difficulty of the game/corpus, from 1 to 10
    """

    question: str
    answer: str
    difficulty: int = 1
    metadata: Optional[Dict[str, Any]] = None
    gpt_response: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_json(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
            "gpt_response": self.gpt_response,
        }
        payload.update(self.extra)
        return payload

    def to_json_str(self) -> str:
        return json.dumps(self.to_json(), ensure_ascii=False)

    @classmethod
    def from_json_str(cls, json_str: str) -> "Data":
        return cls.from_json_dict(json.loads(json_str))

    @classmethod
    def from_json_dict(cls, json_dict: Dict[str, Any]) -> "Data":
        known = {k: json_dict.get(k) for k in ["question", "answer", "difficulty", "metadata"]}
        instance = cls(**known)
        if "gpt_response" in json_dict:
            instance.gpt_response = json_dict["gpt_response"] or ""
        extra = dict(json_dict)
        for k in list(known.keys()) + ["gpt_response"]:
            extra.pop(k, None)
        instance.extra = extra
        return instance

    @classmethod
    def from_jsonl_file(cls, file_path: str) -> List["Data"]:
        data_list: List[Data] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data_list.append(cls.from_json_dict(json.loads(line)))
        return data_list

