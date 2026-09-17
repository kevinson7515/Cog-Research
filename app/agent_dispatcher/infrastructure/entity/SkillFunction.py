

from pydantic import BaseModel


class SkillFunction(BaseModel):
    id: str
    name: str
    description_zh: str
    description_en: str
    parameters: dict | None = None

    def __init__(self, id: str, name: str, description_zh: str, description_en: str, parameters: dict | None = None,
                 **data):
        local = locals()
        fields = self.model_fields
        args_data = dict((k, fields.get(k).default if v is None else v) for k, v in local.items() if k in fields)
        data.update(args_data)
        super().__init__(**data)
