import re
import requests
from typing import Tuple
from bs4 import BeautifulSoup
from cachetools import cached, TTLCache
from markdownify import markdownify as md

from app.common.logger_util import logger

class ContentFetcher:
    """基础网页内容获取器"""

    def __init__(self, proxy: str, timeout: int = 5):
        self.proxy = proxy
        self.timeout = timeout
        logger.info(f'ContentFetcher proxy====> {self.proxy}, timeout====> {self.timeout}')
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    @cached(cache=TTLCache(maxsize=100, ttl=600))
    def fetch(self, url: str) -> Tuple[bool, str]:
        logger.debug(f"开始获取 URL: {url}")
        proxies = {
            "http": self.proxy,
            "https": self.proxy
        }
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout, proxies=proxies)
            response.raise_for_status()
            html = response.content
            logger.debug(f"收到响应: {html[:100]}...")
        except requests.RequestException as e:
            logger.error(f"请求失败: {str(e)}", exc_info=True)
            return False, str(e)

        text = BeautifulSoup(html, 'html.parser').get_text()
        cleaned_text = re.sub(r'\n+', '\n', text)
        logger.info(f"成功获取网页内容，URL: {url}")
        return True, cleaned_text