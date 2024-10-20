from typing import Any, Dict

from src.Databases.DatabaseBase import DatabaseBase
from src.LLM.Pipeline import Pipeline
from src.state_model import GraphState


def find_target_node(state: GraphState, pipeline: Pipeline, database: DatabaseBase) -> Dict[str, Any]:
    print("Finding Target Node...")

    question = state["question"]

    result = pipeline.detect_target_node(content= question, graphdb_nodes=database.list_nodes_and_properties())

    print(result)

    return {"result": result.node_id, "is_relevant": result.is_relevant}
