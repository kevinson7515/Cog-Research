from typing import Optional

from fastapi import APIRouter, Header

from cosight_server.sdk.common.api_result import json_result
from app.common.logger_util import logger
from cosight_server.sdk.entities.chat import Chat

chatRouter = APIRouter()

@chatRouter.get("/chat/list")
def chat_list(client_id: Optional[str] = Header(None)):
    logger.info(f"/chat/list >>>>>>>>>> is called, client_id: {client_id}")
    return json_result(0, "get chat list success", [])


@chatRouter.post("/chat/create")
def chat_create(chat: Chat, client_id: Optional[str] = Header(None), cookie: Optional[str] = Header(None)):
    logger.info(f"/chat/create >>>>>>>>>> is called, client_id: {client_id}, chat: {chat}")
    # assistants = get_cache_config_info(session_manager.get_req_session_id(cookie), "assistants", [])
    # chat.participants = assistants + (chat.participants or [])
    chat.participants = [] + (chat.participants or [])
    chat.showName = False
    chat.showPortrait = False
    chat.showTimestamp = True
    return json_result(0, "success", chat)


@chatRouter.post("/chat/edit")
def chat_edit(chat: Chat, cookie: Optional[str] = Header(None)):
    logger.info(f"/chat/edit >>>>>>>>>> is called, chat: {chat}")
    return json_result(0, "success", chat)
