from typing import Optional

from fastapi import APIRouter, Header
from starlette.responses import Response

from app.common.logger_util import logger
from cosight_server.sdk.services.session_manager import session_manager

userRouter = APIRouter()


@userRouter.get("/deep-research/login")
async def login(
    response: Response,
    cookie: Optional[str] = Header(None),
    referer: Optional[str] = Header(None, alias="Referer")
):
    logger.info(f"login >>>>>>>>>> is called, cookie: {cookie}, referer: {referer}")
    login_res = await session_manager.login(response, cookie, referer)
    return login_res


@userRouter.post("/deep-research/logout")
def logout(response: Response, cookie: Optional[str] = Header(None)):
    logger.info(f"logout >>>>>>>>>> is called")
    logout_res = session_manager.logout(response, cookie)
    return logout_res
