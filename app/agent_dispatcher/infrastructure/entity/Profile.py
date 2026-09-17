

from pydantic import BaseModel

from app.agent_dispatcher.infrastructure.entity.ProfileI18 import ProfileI18


class Profile(BaseModel):
    role: ProfileI18 | None = None
    goal: ProfileI18 | None = None
    instruction: ProfileI18 | None = None
    examples: list[ProfileI18] | None = None
    prompt: ProfileI18 | None = None

    def __init__(self, instruction: ProfileI18 | None = None, role: ProfileI18 | None = None, goal: ProfileI18 | None = None,
                 examples: list[ProfileI18] | None = None, prompt: ProfileI18 | None = None, **data):
        local = locals()
        fields = self.model_fields
        args_data = dict((k, fields.get(k).default if v is None else v) for k, v in local.items() if k in fields)
        data.update(args_data)
        super().__init__(**data)
