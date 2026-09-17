from typing import Optional

from pydantic import BaseModel


class Chat(BaseModel):
    userName: Optional[str] = None
    chatName: Optional[str] = None
    type: Optional[str] = None
    uuid: str
    participants: Optional[list] = None
    messages: Optional[list] = None
    newChatName: Optional[str] = None
    pinMessage: Optional[str] = None
    showPortrait: Optional[bool] = False
    showName: Optional[bool] = False
    showTimestamp: Optional[bool] = False


class RoleInfo(BaseModel):
    name: str
    role: str
    portrait: Optional[str] = None
