from app.cosight.tool.deep_search.common.entity import ModelInfo
from app.cosight.tool.deep_search.model.llm_client import LLMClient


class ModelService():
    process_model = None
    summary_model = None
    deep_research_model = None
    deep_reasoning_models = None
    non_deep_reasoning_models = None
    
    def __init__(self, model_info: ModelInfo):
        self.model_info = model_info
        self._init_models(model_info)

    def _init_models(self, model_info: ModelInfo):
        self.process_model = LLMClient(
            model_name=model_info.get('model_name'), 
            api_url=model_info.get('api_url'), 
            api_key=model_info.get('api_key'), 
            proxy=model_info.get('proxy')
        )
        self.summary_model = self.process_model

        self.deep_reasoning_models = {
            "question_rewriter": self.process_model,
            "knowledge_selector": self.process_model,
            "key_extractor": self.process_model,
            "result_analyst": self.process_model,
            "summary_generator": self.summary_model
        }
        self.non_deep_reasoning_models = {
            "question_rewriter": self.process_model,
            "knowledge_selector": self.process_model,
            "key_extractor": self.process_model,
            "result_analyst": self.process_model,
            "summary_generator": self.process_model
        }
        self.deep_research_model = self.summary_model