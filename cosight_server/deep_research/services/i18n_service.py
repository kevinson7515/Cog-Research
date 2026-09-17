import datetime
import json
import os
from typing import Dict, Any
from contextvars import ContextVar

# 创建上下文变量来存储当前请求的语言设置，确保每次请求都使用同一个语言设置
_current_locale: ContextVar[str] = ContextVar('current_locale', default='zh')

class I18nService:
    TRANSLATIONS = {}
    
    def __init__(self, default_locale: str = 'zh'):
        self.default_locale = default_locale
        self.load_translations_from_file(os.path.join(os.path.dirname(__file__), 'i18n.json'))
        
    def load_translations_from_file(self, file_path: str) -> None:
        with open(file_path, 'r', encoding='utf-8') as f:
            translations = json.load(f)
            self.TRANSLATIONS.update(translations)

    def get_locale(self) -> str:
        """获取当前请求的语言设置"""
        return _current_locale.get()

    def set_locale(self, locale_code: str) -> None:
        """设置当前请求的语言"""
        if locale_code in self.TRANSLATIONS:
            _current_locale.set(locale_code)
        else:
            _current_locale.set(self.default_locale)

    def t(self, key: str, *args: Any, **kwargs: Any) -> str:
        """获取翻译后的文本"""
        current_locale = self.get_locale()
        translations = self.TRANSLATIONS.get(current_locale, self.TRANSLATIONS[self.default_locale])
        text = translations.get(key, key)
        
        if args or kwargs:
            try:
                return text.format(*args, **kwargs)
            except (IndexError, KeyError):
                return text
        return text

# 创建全局实例
i18n = I18nService()
