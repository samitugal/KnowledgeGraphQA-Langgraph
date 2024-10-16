from typing import Any, Dict

from src.LLM.Pipeline import Pipeline
from src.state_model import GraphState


def router_node(state: GraphState, pipeline: Pipeline) -> Dict[str, Any]:
    print("Routing...")

    question = state["question"]

    result = pipeline.router_model(question)

    return {"question": question, "operation_type": result}
