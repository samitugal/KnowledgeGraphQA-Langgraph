from typing import List, TypedDict

from src.output_models import OperationType


class GraphState(TypedDict):
    """
    Represents the state of the graph.

    Attributes:
        question: The question that the user asked.
        generation: LLM Generation
        web_search: Web Search
        documents: List of documents
    """

    question: str
    generation: any
    operation_type: OperationType
    is_relevant: bool
