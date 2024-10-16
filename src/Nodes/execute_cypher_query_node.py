from typing import Any, Dict

from src.output_models import CypherQueryList
from src.Databases.DatabaseBase import DatabaseBase
from src.state_model import GraphState


def execute_cypher_query_node(state: GraphState, database: DatabaseBase) -> Dict[str, Any]:
    print("Executing Cypher Queries...")

    question = state["question"]

    queries = CypherQueryList.model_validate(state["generation"])

    database.flush_all()
    for query in queries.queries:
        database.execute_cypher_query(query.query)

    return {"question": question, "generation": queries}
