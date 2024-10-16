from typing import Any, Dict

from src.Databases.DatabaseBase import DatabaseBase
from src.state_model import GraphState


def graph_db_sanity_check_node(state: GraphState, database: DatabaseBase) -> Dict[str, Any]:
    print("Sanity Checking Graph Database...")

    result = database.sanity_check()

    return {"generation": result}
