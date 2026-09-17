

import time

from pydantic import BaseModel

from app.agent_dispatcher.infrastructure.util.constants import TYPE_REQUEST


class Message(BaseModel):
    content: str
    role: str | None = None
    data: dict | None = None
    create_time: int = int(time.time() * 1000)  # 时间戳,单位毫秒
    type: str = TYPE_REQUEST  # 通知、请求、应答

    def __init__(self, content: str, role: str | None = None, data: dict | None = None, create_time: int = None,
                 type: str = TYPE_REQUEST, **kdata):
        local = locals()
        fields = self.model_fields
        args_data = dict((k, fields.get(k).default if v is None else v) for k, v in local.items() if k in fields)
        kdata.update(args_data)
        super().__init__(**kdata)

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self):
        if self._current_index == 0:
            self._current_index += 1
            return self
        else:
            raise StopIteration

    def to_text(self) -> str:
        return self.content