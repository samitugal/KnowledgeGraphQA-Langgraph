from typing import Any, Dict

from src.LLM.Pipeline import Pipeline
from src.state_model import GraphState

def generation_node(state: GraphState, pipeline: Pipeline) -> Dict[str, Any]:
    print("Generating Results...")

    question = state["question"]
    results = state["result"]

    result = pipeline.generation_model(context=results, question=question)

    return {"result": result, "question": question}


