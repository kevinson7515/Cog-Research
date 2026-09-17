from app.common.logger_util import logger
class TerminateToolkit:
    r"""A class representing a toolkit for terminating interactions when the request is met OR if the assistant cannot proceed further with the task."""

    def __init__(self):
        pass

    def terminate(self, status: str, reason: str) -> str:
        r"""Finish the current execution.

        Args:
            status (str): The finish status of the interaction.
            reason (str): The finish reason of the interaction.

        Returns:
            str: The termination message.
        """
        logger.info(f"Terminating interaction with status: {status}, with reason: {reason}")
        return f"The interaction has been completed with status: {status}, with reason: {reason}"
