# Copyright 2025 ZTE Corporation.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

from dotenv import load_dotenv

load_dotenv()
import os
from openai import OpenAI
import httpx
import base64
import asyncio
from urllib.parse import urlparse

from app.common.logger_util import logger


class AudioTool:
    def __init__(self, llm_config):
        self.llm_config = llm_config

    name: str = "Audio Tool"
    description: str = (
        "This tool uses OpenAI's Audio API to describe the contents of an audio."
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
            self._client = OpenAI(http_client=self._http_client(), **llm_config)
        return self._client

    def encode_audio(self, audio_path):
        with open(audio_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")

    def get_audio_extension(self, url):
        parsed = urlparse(url)
        path = parsed.path
        return os.path.splitext(path)[1].lower()

    async def audio_recognition(self, audio_path, task_prompt):
        if audio_path.startswith('http://') or audio_path.startswith('https://'):
            audio_url = audio_path
            audio_format = self.get_audio_extension(audio_path)
        else:
            base64_audio = self.encode_audio(audio_path)
            audio_url = f"data:;base64,{base64_audio}"
            audio_format = os.path.splitext(audio_path)[-1]

        api_params = {
            "extra_headers": {'Content-Type': 'application/json',
                              'Authorization': 'Bearer %s' % self.llm_config['api_key']},
            "model": self.llm_config['model'],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_url,
                                "format": audio_format,
                            },
                        },
                        {"type": "text", "text": task_prompt},
                    ],
                },
            ],
            "modalities": ["text", "audio"],
            "audio": {"voice": "Cherry", "format": "wav"},
            "stream": True,
            "stream_options": {"include_usage": True},
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
                    except Exception:
                        pass
        return full_response

    def speech_to_text(self, audio_path: str, task_prompt: str, ):
        logger.info(f"Using Tool: {self.name}, audio_path: {audio_path}, task_prompt: {task_prompt}")
        return asyncio.run(self.audio_recognition(audio_path, task_prompt))
