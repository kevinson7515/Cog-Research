from app.agent_dispatcher.infrastructure.entity.SkillFunction import SkillFunction

def terminate_skill():
    return {
        'skill_name': 'terminate',
        'skill_type': "function",
        'display_name_zh': '终止交互',
        'display_name_en': 'Terminate Interaction',
        'description_zh': '当请求完成或无法继续任务时终止交互',
        'description_en': 'Terminate interaction when request is met or task cannot proceed further',
        'semantic_apis': ["api_termination"],
        'function': SkillFunction(
            id='5c44f9ad-be5c-4e6c-a9d8-1426b23828a8',
            name='app.cosight.planner.terminate_toolkit.TerminateToolkit.terminate',
            description_zh='终止当前交互并返回状态和原因',
            description_en='Terminate current interaction and return status and reason',
            parameters={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description_zh": "交互的终止状态",
                        "description_en": "Termination status of the interaction"
                    },
                    "reason": {
                        "type": "string",
                        "description_zh": "交互的终止原因",
                        "description_en": "Termination reason of the interaction"
                    }
                },
                "required": ["status", "reason"]
            }
        )
    }
