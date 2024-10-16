from abc import ABC, abstractmethod

from ..output_models import *


class LLMAbstractBase(ABC):

    @abstractmethod
    def generate_kg_query(self, content: str) -> CypherQueryList:
        pass

    @abstractmethod
    def router_model(self, user_request: str) -> RouterModelOutput:
        pass
