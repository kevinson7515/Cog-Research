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

import asyncio
import json
import httpx
from typing import Optional, List

from httpx import ReadTimeout
from lagent.actions.bing_browser import DuckDuckGoSearch
from app.cosight.tool.deep_search.common.entity import SearchSource
from config.config import get_serper_config
from app.common.logger_util import logger


class SerperSearch(DuckDuckGoSearch):
    """Serper搜索引擎实现"""
    def __init__(self,
                proxy: Optional[str] = None,
                topk: int = 10,
                black_list: Optional[List[str]] = None,
                web_source: Optional[SearchSource] = None,
                **kwargs):
        super().__init__(**kwargs)
        self.api_key = get_serper_config()
        self.proxy = proxy
        self.topk = topk
        self.black_list = black_list or []
        self.api_url = "https://google.serper.dev/images"
        self.timeout = kwargs.get("timeout", 5)
        self.include_domains = []
        if web_source:
            urls = web_source.get("config", {}).get("urls", [])
            self.include_domains = urls

    def _call_ddgs(self, query: str, **kwargs) -> dict:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(self._async_call_ddgs(query, **kwargs))
            return response
        finally:
            loop.close()

    async def _async_call_ddgs(self, query: str, **kwargs) -> dict:
        """实现 Serper 图像检索接口"""
        logger.info(f"开始 Serper 图像检索，查询内容: {query}")

        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": self.api_key,
        }

        data = {
            "q": query,
            "num": self.topk,
        }

        # 如果你们项目有需要，也可以把地区参数放开
        # data["gl"] = kwargs.get("gl", "cn")
        # data["hl"] = kwargs.get("hl", "zh-cn")

        try:
            logger.debug(f"正在连接 Serper API，URL: {self.api_url}")
            async with httpx.AsyncClient(proxy=self.proxy, timeout=self.timeout) as client:
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    json=data
                )
                response.raise_for_status()
                response_data = response.json()

                images = response_data.get("images", [])
                if not isinstance(images, list):
                    return []

                # 统一成和你 TavilySearch 类似的中间结构
                result = []
                for item in images:
                    image_url = item.get("imageUrl") or item.get("url") or ""
                    page_url = item.get("link") or ""
                    domain = item.get("domain") or ""
                    if not image_url:
                        continue

                    result.append({
                        "title": item.get("title", ""),
                        "image_url": image_url,
                        "page_url": page_url,
                        "thumbnail_url": item.get("thumbnailUrl", ""),
                        "source": item.get("source", ""),
                        "domain": domain,
                        "google_url": item.get("googleUrl", ""),
                        "width": item.get("imageWidth", 0),
                        "height": item.get("imageHeight", 0),
                        "thumbnail_width": item.get("thumbnailWidth", 0),
                        "thumbnail_height": item.get("thumbnailHeight", 0),
                        "position": item.get("position", 0),
                        "score": 1.0 / max(item.get("position", 1), 1),
                    })

                logger.info(f"Serper 图像检索成功，返回结果数量: {len(result)}")
                return {
                    "content": result,
                    "images": images
                }

        except httpx.TimeoutException as e:
            logger.error(f"Serper HTTP请求超时: {str(e)}", exc_info=True)
            return []
        except httpx.HTTPError as e:
            logger.error(
                f"Serper HTTP请求失败: {str(e)}, 状态码: {getattr(e.response, 'status_code', 'N/A')}, "
                f"响应内容: {getattr(e.response, 'text', 'N/A')}",
                exc_info=True
            )
            return []
        except Exception as e:
            logger.error(f"Serper 图像检索失败，详细错误: {str(e)}", exc_info=True)
            return []

    def _parse_response(self, response: dict) -> dict:
        raw_results = []
        for item in response.get("content", []):
            raw_results.append((
                item.get("page_url", ""),
                item.get("title", ""),
                item.get("title", ""),
                item.get("score", 0),
                item
            ))

        results = self._filter_results(raw_results)
        images = response.get("images", [])
        return {
            "content": results,
            "images": images
        }

    def _filter_results(self, results: List[tuple]) -> dict:
        """
        过滤并重组结果。

        results 结构：
        (page_url, snippet, title, score, raw_item)
        """
        filtered_results = {}
        count = 0

        for item in results:
            if len(item) == 5:
                page_url, snippet, title, score, raw_item = item
            else:
                page_url, snippet, title, score = item
                raw_item = {}

            url_to_check = page_url or raw_item.get("image_url", "")
            if not url_to_check:
                continue

            if any(domain in url_to_check for domain in self.black_list):
                continue

            # 图像检索里通常更关键的是 image_url，这里默认保留
            filtered_results[count] = {
                "url": raw_item.get("image_url", url_to_check),
                "page_url": page_url,
                "thumb_url": raw_item.get("thumbnail_url", ""),
                "summ": json.dumps(snippet, ensure_ascii=False)[1:-1] if snippet else "",
                "title": title,
                "source": raw_item.get("source", ""),
                "domain": raw_item.get("domain", ""),
                "google_url": raw_item.get("google_url", ""),
                "width": raw_item.get("width", 0),
                "height": raw_item.get("height", 0),
                "score": score
            }
            count += 1
            if count >= self.topk:
                break

        return filtered_results