
class ZaeFrameworkException(Exception):
    def __init__(self, error_code, message):
        super().__init__(f"Error {error_code}: {message}")
        self.error_code = error_code


class NaeFrameworkException(ZaeFrameworkException):
    def __init__(self, error_code, message):
        super().__init__(error_code, message)


# Started by AICoder, pid:025a4a02719e4bd69a769bc7b22fabad
class FunctionTimeoutException(NaeFrameworkException):
    def __init__(self, error_code, message):
        super().__init__(error_code, message)
# Ended by AICoder, pid:025a4a02719e4bd69a769bc7b22fabad
