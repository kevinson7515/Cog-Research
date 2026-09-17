from dotenv import load_dotenv

load_dotenv()
import os
from openai import OpenAI
import httpx
import base64
import asyncio

from app.common.logger_util import logger


class VisionTool():
    def __init__(self, llm_config):
        self.llm_config = llm_config
        self.max_tokens = int(os.environ.get("VISION_TOOL_MAX_TOKENS", str(llm_config.get("max_tokens") or 1024)))
        self.max_response_chars = int(os.environ.get("VISION_TOOL_MAX_RESPONSE_CHARS", "12000"))

    name: str = "Vision Tool"
    description: str = (
        "This tool uses OpenAI's Vision API to describe the contents of an Image."
    )
    _client: OpenAI = None

    def _http_client(self) -> httpx.Client:
        timeout_seconds = float(os.environ.get("LLM_TIMEOUT", "180"))
        http_client_kwargs = {
            "trust_env": False,
            "timeout": httpx.Timeout(
                connect=30.0,
                read=timeout_seconds,
                write=30.0,
                pool=10.0,
            ),
        }
        proxy = self.llm_config.get("proxy")
        if proxy:
            http_client_kwargs["proxy"] = proxy
        return httpx.Client(**http_client_kwargs)

    def _extra_body(self) -> dict:
        base_url = str(self.llm_config.get("base_url") or "").lower()
        model = str(self.llm_config.get("model") or "").lower()
        thinking_enabled = bool(self.llm_config.get("thinking_mode"))
        if "dashscope.aliyuncs.com" in base_url and "qwen" in model and not thinking_enabled:
            return {"enable_thinking": False}
        return {}

    @property
    def client(self) -> OpenAI:
        llm_config = {"api_key": self.llm_config['api_key'],
                      "base_url": self.llm_config['base_url']
                      }
        """Cached ChatOpenAI client instance."""
        if self._client is None:
            # Avoid broken cluster env (SSL_CERT_FILE, proxy vars) unless VISION_PROXY is explicit.
            self._client = OpenAI(http_client=self._http_client(), **llm_config)
        return self._client

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    async def _run(self, image_path_url, task_prompt):
        if image_path_url.startswith('http://') or image_path_url.startswith('https://'):
            img_url = image_path_url
        else:
            base64_image = self.encode_image(image_path_url)
            img_url = f"data:image/png;base64,{base64_image}"

        api_params = {
            "extra_headers": {'Content-Type': 'application/json',
                              'Authorization': 'Bearer %s' % self.llm_config['api_key']},
            "model": self.llm_config['model'],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": img_url},
                        },
                        {"type": "text", "text": task_prompt},
                    ],
                },
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": self.max_tokens,
        }
        extra_body = self._extra_body()
        if extra_body:
            api_params["extra_body"] = extra_body

        completion = self.client.chat.completions.create(**api_params)
        full_response = ""
        for chunk in completion:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "audio") and delta.audio:
                    try:
                        if delta.audio['transcript']:
                            full_response += delta.audio['transcript']
                    except Exception:
                        pass
                if hasattr(delta, "content") and delta.content:
                    try:
                        full_response += delta.content
                        if len(full_response) >= self.max_response_chars:
                            logger.warning(
                                f"Vision response reached {self.max_response_chars} chars; truncating to avoid oversized context"
                            )
                            return full_response[:self.max_response_chars]
                    except Exception:
                        pass
        return full_response

    def ask_question_about_image(self, image_path_url, task_prompt):
        logger.info(f"Using Tool: {self.name}, image_path_url: {image_path_url}, task_prompt: {task_prompt}")
        return asyncio.run(self._run(image_path_url, task_prompt))
