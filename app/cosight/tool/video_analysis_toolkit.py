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
from app.common.logger_util import logger


class VideoTool:
    def __init__(self, llm_config):
        self.llm_config = llm_config

    name: str = "Video Tool"
    description: str = (
        "This tool uses OpenAI's Video API to describe the contents of an video."
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

    def encode_video(self, video_path):
        with open(video_path, "rb") as video_file:
            return base64.b64encode(video_file.read()).decode("utf-8")

    async def video_analy(self, video_path: str, question: str):
        if video_path.startswith('http://') or video_path.startswith('https://'):
            video_url = video_path
        else:
            base64_video = self.encode_video(video_path)
            video_url = f"data:;base64,{base64_video}"

        api_params = {
            "extra_headers": {'Content-Type': 'application/json',
                              'Authorization': 'Bearer %s' % self.llm_config['api_key']},
            "model": self.llm_config['model'],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": video_url},
                        },
                        {"type": "text", "text": question},
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

    def ask_question_about_video(self, video_path: str, question: str, ):
        logger.info(f"Using Tool: {self.name}, video_path: {video_path}, question: {question}")
        return asyncio.run(self.video_analy(video_path, question))
