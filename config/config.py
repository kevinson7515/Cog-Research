import os
from dotenv import load_dotenv
from typing import Optional

# 加载.env文件到环境变量：
# - 默认不覆盖进程环境变量，允许通过启动脚本/容器 `export` 来覆盖 .env 中的值（便于本地 vLLM 场景）
# - 如需强制以 .env 为准，可设置 `DOTENV_OVERRIDE=true`
dotenv_override = os.environ.get("DOTENV_OVERRIDE", "").strip().lower() in ("1", "true", "yes", "enabled", "on")
load_dotenv(override=dotenv_override)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return default
    return value


def _default_api_key() -> str:
    # Local OpenAI-compatible servers such as vLLM require a non-empty client
    # api_key value, but usually do not validate it.
    return _env("API_KEY") or _env("OPENAI_API_KEY") or "EMPTY"


def _default_base_url() -> Optional[str]:
    port = _env("VLLM_PORT", "8001")
    return _env("API_BASE_URL") or _env("OPENAI_BASE_URL") or f"http://127.0.0.1:{port}/v1"


def _default_model_name() -> Optional[str]:
    return _env("MODEL_NAME") or _env("SERVED_MODEL_NAME")


# ========== 大模型配置 ==========
def get_model_config() -> dict[str, Optional[str | int | float | bool]]:
    """获取API配置"""
    max_tokens = os.environ.get("MAX_TOKENS")
    temperature = os.environ.get("TEMPERATURE")
    thinking_mode = os.environ.get("THINKING_MODE", "").strip().lower()
    api_key = _default_api_key()
    os.environ["OPENAI_API_KEY"] = api_key
    return {
        "api_key": api_key,
        "base_url": _default_base_url(),
        "model": _default_model_name(),
        "max_tokens": int(max_tokens) if max_tokens and max_tokens.strip() else None,
        "temperature": float(temperature) if temperature and temperature.strip() else None,
        "proxy": os.environ.get("PROXY"),
        "thinking_mode": thinking_mode in ("1", "true", "yes", "enabled", "on")
    }


# ========== 规划大模型配置 ==========
def get_plan_model_config() -> dict[str, Optional[str | int | float | bool]]:
    """获取Plan专用API配置，如果缺少配置则退回默认"""
    plan_api_key = os.environ.get("PLAN_API_KEY")
    plan_base_url = os.environ.get("PLAN_API_BASE_URL")
    model_name = os.environ.get("PLAN_MODEL_NAME")

    # 检查三个字段是否都存在且非空
    if not (plan_api_key and plan_base_url and model_name):
        return get_model_config()

    max_tokens = os.environ.get("PLAN_MAX_TOKENS")
    temperature = os.environ.get("PLAN_TEMPERATURE")
    thinking_mode = os.environ.get("PLAN_THINKING_MODE", "").strip().lower()

    return {
        "api_key": plan_api_key,
        "base_url": plan_base_url,
        "model": model_name,
        "max_tokens": int(max_tokens) if max_tokens and max_tokens.strip() else None,
        "temperature": float(temperature) if temperature and temperature.strip() else None,
        "proxy": os.environ.get("PLAN_PROXY"),
        "thinking_mode": thinking_mode in ("1", "true", "yes", "enabled", "on") if thinking_mode else get_model_config().get("thinking_mode", False)
    }


# ========== 执行大模型配置 ==========
def get_act_model_config() -> dict[str, Optional[str | int | float | bool]]:
    """获取Act专用API配置，如果缺少配置则退回默认"""
    act_api_key = os.environ.get("ACT_API_KEY")
    act_base_url = os.environ.get("ACT_API_BASE_URL")
    model_name = os.environ.get("ACT_MODEL_NAME")

    # 检查三个字段是否都存在且非空
    if not (act_api_key and act_base_url and model_name):
        return get_model_config()

    max_tokens = os.environ.get("ACT_MAX_TOKENS")
    temperature = os.environ.get("ACT_TEMPERATURE")
    thinking_mode = os.environ.get("ACT_THINKING_MODE", "").strip().lower()

    return {
        "api_key": act_api_key,
        "base_url": act_base_url,
        "model": model_name,
        "max_tokens": int(max_tokens) if max_tokens and max_tokens.strip() else None,
        "temperature": float(temperature) if temperature and temperature.strip() else None,
        "proxy": os.environ.get("ACT_PROXY"),
        "thinking_mode": thinking_mode in ("1", "true", "yes", "enabled", "on") if thinking_mode else get_model_config().get("thinking_mode", False)
    }


# ========== 工具大模型配置 ==========
def get_tool_model_config() -> dict[str, Optional[str | int | float | bool]]:
    """获取Tool专用API配置，如果缺少配置则退回默认"""
    tool_api_key = os.environ.get("TOOL_API_KEY")
    tool_base_url = os.environ.get("TOOL_API_BASE_URL")
    model_name = os.environ.get("TOOL_MODEL_NAME")

    # 检查三个字段是否都存在且非空
    if not (tool_api_key and tool_base_url and model_name):
        return get_model_config()

    max_tokens = os.environ.get("TOOL_MAX_TOKENS")
    temperature = os.environ.get("TOOL_TEMPERATURE")
    thinking_mode = os.environ.get("TOOL_THINKING_MODE", "").strip().lower()

    return {
        "api_key": tool_api_key,
        "base_url": tool_base_url,
        "model": model_name,
        "max_tokens": int(max_tokens) if max_tokens and max_tokens.strip() else None,
        "temperature": float(temperature) if temperature and temperature.strip() else None,
        "proxy": os.environ.get("TOOL_PROXY"),
        "thinking_mode": thinking_mode in ("1", "true", "yes", "enabled", "on") if thinking_mode else get_model_config().get("thinking_mode", False)
    }


# ========== 多模态大模型配置 ==========
def get_vision_model_config() -> dict[str, Optional[str | int | float | bool]]:
    """获取Vision专用API配置，如果缺少配置则退回默认"""
    vision_api_key = os.environ.get("VISION_API_KEY")
    vision_base_url = os.environ.get("VISION_API_BASE_URL")
    model_name = os.environ.get("VISION_MODEL_NAME")

    # 检查三个字段是否都存在且非空
    if not (vision_api_key and vision_base_url and model_name):
        return get_model_config()

    max_tokens = os.environ.get("VISION_MAX_TOKENS")
    temperature = os.environ.get("VISION_TEMPERATURE")
    thinking_mode = os.environ.get("VISION_THINKING_MODE", "").strip().lower()

    return {
        "api_key": vision_api_key,
        "base_url": vision_base_url,
        "model": model_name,
        "max_tokens": int(max_tokens) if max_tokens and max_tokens.strip() else None,
        "temperature": float(temperature) if temperature and temperature.strip() else None,
        "proxy": os.environ.get("VISION_PROXY"),
        "thinking_mode": thinking_mode in ("1", "true", "yes", "enabled", "on") if thinking_mode else get_model_config().get("thinking_mode", False)
    }


# ========== 可信信息分析大模型配置 ==========
def get_credibility_model_config() -> dict[str, Optional[str | int | float | bool]]:
    """获取可信信息分析专用API配置，如果缺少配置则退回默认"""
    credibility_api_key = os.environ.get("CREDIBILITY_API_KEY")
    credibility_base_url = os.environ.get("CREDIBILITY_API_BASE_URL")
    model_name = os.environ.get("CREDIBILITY_MODEL_NAME")

    # 检查三个字段是否都存在且非空
    if not (credibility_api_key and credibility_base_url and model_name):
        return get_model_config()

    max_tokens = os.environ.get("CREDIBILITY_MAX_TOKENS")
    temperature = os.environ.get("CREDIBILITY_TEMPERATURE")
    thinking_mode = os.environ.get("CREDIBILITY_THINKING_MODE", "").strip().lower()

    return {
        "api_key": credibility_api_key,
        "base_url": credibility_base_url,
        "model": model_name,
        "max_tokens": int(max_tokens) if max_tokens and max_tokens.strip() else None,
        "temperature": float(temperature) if temperature and temperature.strip() else None,
        "proxy": os.environ.get("CREDIBILITY_PROXY"),
        "thinking_mode": thinking_mode in ("1", "true", "yes", "enabled", "on") if thinking_mode else get_model_config().get("thinking_mode", False)
    }


# ========== 浏览器自动化大模型配置 ==========
def get_browser_model_config() -> dict[str, Optional[str | int | float | bool]]:
    """获取浏览器自动化专用API配置，如果缺少配置则退回默认"""
    browser_api_key = os.environ.get("BROWSER_API_KEY")
    browser_base_url = os.environ.get("BROWSER_API_BASE_URL")
    model_name = os.environ.get("BROWSER_MODEL_NAME")

    # 检查三个字段是否都存在且非空
    if not (browser_api_key and browser_base_url and model_name):
        return get_model_config()

    max_tokens = os.environ.get("BROWSER_MAX_TOKENS")
    temperature = os.environ.get("BROWSER_TEMPERATURE")
    thinking_mode = os.environ.get("BROWSER_THINKING_MODE", "").strip().lower()

    return {
        "api_key": browser_api_key,
        "base_url": browser_base_url,
        "model": model_name,
        "max_tokens": int(max_tokens) if max_tokens and max_tokens.strip() else None,
        "temperature": float(temperature) if temperature and temperature.strip() else None,
        "proxy": os.environ.get("BROWSER_PROXY"),
        "thinking_mode": thinking_mode in ("1", "true", "yes", "enabled", "on") if thinking_mode else get_model_config().get("thinking_mode", False)
    }


# ========== 工具配置 ==========
def get_tavily_config() -> Optional[str]:
    """获取tavily兼容API配置"""
    return os.environ.get("TAVILY_API_KEY")


def get_serper_config() -> Optional[str]:
    """获取serper兼容API配置"""
    return os.environ.get("SERPER_API_KEY")


def get_turbo_mode() -> bool:
    """获取急速模式配置"""
    turbo_mode = os.environ.get("TURBO_MODE", "").strip().lower()
    return turbo_mode in ("1", "true", "yes", "enabled", "on")


def validate_config(config: dict) -> bool:
    """验证必要配置是否存在"""
    if not config.get("api_key"):
        raise ValueError("OPENAI_COMPATIBILITY_API_KEY 未配置")
    return True
