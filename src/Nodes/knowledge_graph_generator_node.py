from typing import Any, Dict

from src.LLM.Pipeline import Pipeline
from src.state_model import GraphState


def knowledge_graph_generator_node(state: GraphState, pipeline: Pipeline) -> Dict[str, Any]:
    print("Generating Knowledge Graph...")

    question = state["question"]

    result = pipeline.generate_kg_query(question)

    return {"question": question, "generation": result}
