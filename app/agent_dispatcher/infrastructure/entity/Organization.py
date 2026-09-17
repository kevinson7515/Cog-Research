

#!/usr/bin/env python
# coding=utf-8

from pydantic import BaseModel


class Organization(BaseModel):
    organization_name: str
    display_name_zh: str
    display_name_en: str
    icon: str | None = None
    superior: str | None = None
    reserved_map: dict | None = None

    def __init__(self, organization_name: str, display_name_zh: str, display_name_en: str, icon: str | None = None,
                 superior: str | None = None, reserved_map: dict | None = None, **data):
        local = locals()
        fields = self.model_fields
        args_data = dict((k, fields.get(k).default if v is None else v) for k, v in local.items() if k in fields)
        data.update(args_data)
        super().__init__(**data)
