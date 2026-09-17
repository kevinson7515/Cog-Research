

from pydantic import BaseModel


class Conversation(BaseModel):
    speaker: str
    content: str
    listener: str

    def __init__(self, speaker: str, content: str, listener: str, **data):
        local = locals()
        fields = self.model_fields
        args_data = dict((k, fields.get(k).default if v is None else v) for k, v in local.items() if k in fields)
        data.update(args_data)
        super().__init__(**data)


class ConversationHistory(BaseModel):
    history: list[Conversation] = []

    def __init__(self, history: list[Conversation] = None, **data):
        local = locals()
        fields = self.model_fields
        args_data = dict((k, fields.get(k).default if v is None else v) for k, v in local.items() if k in fields)
        data.update(args_data)
        super().__init__(**data)

    def append(self, conversation: Conversation):
        self.history.append(conversation)
