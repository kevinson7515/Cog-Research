

from pydantic import BaseModel


class ProfileI18(BaseModel):
    zh: str
    en: str
    title: str | None = None

    def __init__(self, zh: str, en: str, title: str | None = None, **data):
        local = locals()
        fields = self.model_fields
        args_data = dict((k, fields.get(k).default if v is None else v) for k, v in local.items() if k in fields)
        data.update(args_data)
        super().__init__(**data)
