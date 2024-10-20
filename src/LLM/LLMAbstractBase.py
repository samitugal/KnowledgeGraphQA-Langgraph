from abc import ABC, abstractmethod

from ..output_models import *


class LLMAbstractBase(ABC):

    @abstractmethod
    def generate_kg_query(self, content: str) -> CypherQueryList:
        pass

    @abstractmethod
    def router_model(self, user_request: str) -> RouterModelOutput:
        pass

    @abstractmethod
    def answer_question_model(self, context: str, question: str) -> AnswerQuestionModelOutput:
        pass

    @abstractmethod
    def detect_target_node(self, content: str, graphdb_nodes: List[Dict[str, Any]]) -> NodeDetectionModelOutput:
        pass