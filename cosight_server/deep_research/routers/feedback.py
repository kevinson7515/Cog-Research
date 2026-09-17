from fastapi import APIRouter

from cosight_server.sdk.common.api_result import json_result
from app.common.logger_util import logger

feedbackRouter = APIRouter()


@feedbackRouter.get("/feedback/reasons")
def feedback_reasons(lang: str):
    logger.info(f"feedback_reasons >>>>>>>>>>>>>>>>> is called, lang: {lang}")
    return json_result(0, "", [])
