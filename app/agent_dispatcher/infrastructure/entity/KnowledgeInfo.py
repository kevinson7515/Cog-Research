

from typing import Any

from pydantic import BaseModel


class KnowledgeInfo(BaseModel):
    name: str
    priority: int
    display_name_zh: str
    display_name_en: str

    def __init__(self, name: str, priority: int, display_name_zh: str, display_name_en: str, **data: Any):
        local = locals()
        fields = self.model_fields
        args_data = dict((k, fields.get(k).default if v is None else v) for k, v in local.items() if k in fields)
        data.update(args_data)
        super().__init__(**data)
