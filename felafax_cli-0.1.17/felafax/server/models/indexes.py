from pydantic import BaseModel
from typing import Dict, Optional
from ..handlers.base import DictBaseModel

class TokenIndex(DictBaseModel):
    """Token to user_id mapping index"""
    tokens: Dict[str, str] = {}  # Maps token -> user_id

    def get_value(self, key: str) -> Optional[str]:
        return self.tokens.get(key)

    def set_value(self, key: str, value: str) -> None:
        self.tokens[key] = value

    def remove_value(self, key: str) -> None:
        if key in self.tokens:
            del self.tokens[key]

class IndexPaths:
    @staticmethod
    def token_index_path() -> str:
        return "indexes/tokens.json"
