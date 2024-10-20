from typing import Any, Dict, List

from ..output_models import *
from .config_defs import LLMMainConfig, LLMTag
from .LLMBase import LLMBase


class Pipeline:
    def __init__(self, config: LLMMainConfig, llm: LLMBase):
        self.config = config
        self.llm = llm

    @staticmethod
    def new_instance_from_config(config: LLMMainConfig) -> "Pipeline":
        from .Bedrock import Bedrock
        from .OpenAI import OpenAI

        match config.llm.llm_tag:
            case LLMTag.BEDROCK:
                return Pipeline(config, Bedrock(config))
            case LLMTag.OPENAI:
                return Pipeline(config, OpenAI(config))
            case _:
                raise ValueError("Invalid LLM tag")

    def generate_kg_query(self, content: str) -> CypherQueryList:
        return self.llm.generate_kg_query(content)

    def detect_target_node(
        self, content: str, graphdb_nodes: list
    ) -> NodeDetectionModelOutput:
        return self.llm.detect_target_node(content, graphdb_nodes)

    def router_model(self, user_request: str) -> RouterModelOutput:
        return self.llm.router_model(user_request)
    
    def generation_model(self, context: str, question: str) -> GenerationModelOutput:
        return self.llm.generation_model(context, question)

    def answer_question_model(self, context: str, question: str) -> AnswerQuestionModelOutput:
        return self.llm.answer_question_model(context, question)