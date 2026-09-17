
from pydantic import BaseModel

from app.agent_dispatcher.infrastructure.entity.AgentTemplate import AgentTemplate


class AgentInstance(BaseModel):
    instance_id: str
    instance_name: str
    template_name: str
    template_version: str
    display_name_zh: str
    display_name_en: str
    description_zh: str
    description_en: str
    service_name: str
    service_version: str
    template: AgentTemplate | None = None

    def __init__(self, instance_id: str, instance_name: str, template_name: str, template_version: str,
                 display_name_zh: str, display_name_en: str, description_zh: str, description_en: str,
                 service_name: str, service_version: str, template: AgentTemplate | None, **data):
        local = locals()
        fields = self.model_fields
        args_data = dict((k, fields.get(k).default if v is None else v) for k, v in local.items() if k in fields)
        data.update(args_data)
        super().__init__(**data)