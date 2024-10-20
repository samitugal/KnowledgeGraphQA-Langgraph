from typing import Any, Dict

from src.Databases.DatabaseBase import DatabaseBase
from src.LLM.Pipeline import Pipeline
from src.state_model import GraphState


def answer_question_node(state: GraphState, pipeline: Pipeline, database: DatabaseBase) -> Dict[str, Any]:
    print("Answering Question...")

    degree_count = 1
    previous_node_schema = None

    question = state["question"]
    related_node_id = state["result"]

    related_nodes = database.list_n_degree_nodes(related_node_id, degree_count)

    result = pipeline.answer_question_model(context=related_nodes, question=question)

    while not result.success:
        if related_nodes == previous_node_schema:
            return {"result": "I cannot answer this question.", "question": question}
        else:
            degree_count += 1
            previous_node_schema = related_nodes
            related_nodes = database.list_n_degree_nodes(related_node_id, degree_count)
            result = pipeline.answer_question_model(context=related_nodes, question=question)
            print(result)

    return {"result": result, "question": question}
