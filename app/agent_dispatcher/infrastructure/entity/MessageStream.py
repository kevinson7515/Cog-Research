

from typing import Generator, Any


class MessageStream:
    def __init__(self, generator: Generator):
        self.generator = generator  # 原始生成器
        if not isinstance(generator,Generator):
            #如果不是生成器，变成只有一个元素的生成器
            self.generator=(x for x in [generator])
        self._cache = []  # 缓存已产生的值
        self._current_index = 0  # 当前迭代位置

    def __iter__(self):
        return self

    def __next__(self) -> Any:
        if self._current_index < len(self._cache):
            value = self._cache[self._current_index]
            self._current_index += 1
            return value
        try:
            next_value = next(self.generator)
            self._cache.append(next_value)
            self._current_index += 1
            return next_value
        except StopIteration:
            self._current_index = 0  # 重置索引以便支持重新开始迭代
            raise StopIteration

    def to_text(self) -> str:
        # 如果缓存为空，则先尝试填充缓存
        if not self._cache:
            for item in self.generator:
                self._cache.append(item)
        # 使用存储的值来拼接字符串
        return ''.join(str(x) for x in self._cache)