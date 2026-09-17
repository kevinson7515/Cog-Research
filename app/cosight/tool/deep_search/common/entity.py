from typing import Any, Dict, List, Optional, TypedDict, Union

class SearchSourceType:
    ICENTER = "ZTEICenterDocument"
    RAG = "RAGKnowledgeLibrary"
    WEB = "ManusWebSearch"

class SearchSource(TypedDict):
    id: Union[int, str]
    type: SearchSourceType
    name: str
    sub_name: Optional[str]
    description: str
    owner: Optional[str]
    source_from: Optional[str]
    config: Optional[Dict]

class ContentItem(TypedDict):
    type: str
    value: str
    
class SearchParams(TypedDict):
    content: List[ContentItem]
    history: List[ContentItem]
    sessionInfo: Any
    stream: bool
    contentProperties: str

class SearchResult(TypedDict):
    analysis: str
    summary: str

class ModelInfo(TypedDict):
    base_url: Optional[str]
    api_url: str
    api_key: str
    model_name: str
    proxy: str
    
class WebSearchInfo(TypedDict):
    proxy: str
    api_key: str

