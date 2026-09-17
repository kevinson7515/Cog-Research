from typing import Optional, Any

from pydantic import BaseModel


class ConfigInfo(BaseModel):
    selectedModel: Optional[int] = None


class ConfigSetInfo(BaseModel):
    key: str
    value: Optional[Any] = None
