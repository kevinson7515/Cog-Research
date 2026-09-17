# 返回格式
def json_result(code=None, msg=None, data=None):
    return {
        "code": code,
        "msg": msg,
        "data": data,
    }
