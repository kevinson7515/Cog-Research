

from pydantic import BaseModel

from app.agent_dispatcher.infrastructure.entity.SkillFunction import SkillFunction


# class SkillType(str, Enum):
#     skill = "skill"
#     function = "function"
#     workflow = "workflow"
#     semantic_workflow = "semantic_workflow"


class Skill(BaseModel):
    skill_name: str
    skill_type: str  # SkillType
    display_name_zh: str
    display_name_en: str
    description_zh: str | None = None
    description_en: str | None = None
    semantic_apis: list[str] = []
    function: SkillFunction | None = None
    workflow: dict | None = None
    is_visible: bool = True
    reserved_map: dict | None = None
    mcp_server_config: dict | None = None

    def __init__(self, skill_name: str, skill_type: str, display_name_zh: str, display_name_en: str,
                 description_zh: str | None = None, description_en: str | None = None, semantic_apis: list[str] = None,
                 function: SkillFunction | None = None, workflow: dict | None = None, is_visible: bool = True,
                 reserved_map: dict | None = None, mcp_server_config: dict | None = None, **data):
        local = locals()
        fields = self.model_fields
        args_data = dict((k, fields.get(k).default if v is None else v) for k, v in local.items() if k in fields)
        data.update(args_data)
        super().__init__(**data)

    # @model_validator(mode="after")
    # def check_skill_type(self):
    #     if self.skill_type == SkillType.skill and not (self.description_zh and self.description_en and self.semantic_apis):
    #         raise ValueError(
    #             'When the skill_type field is set to skill, the semantic_apis, description_zh and description_en fields are required.')
    #     if self.skill_type == SkillType.function and self.function is None:
    #         raise ValueError('When the skill_type field is set to function, the function field is required.')
    #     if self.skill_type == SkillType.workflow and self.workflow is None:
    #         raise ValueError('When the skill_type field is set to workflow, the workflow field is required.')
