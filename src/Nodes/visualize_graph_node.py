from typing import Any, Dict

from src.Databases.DatabaseBase import DatabaseBase
from src.Tools.graph_visualizer import visualize_graph
from src.state_model import GraphState


def visualize_graph_node(state: GraphState, database: DatabaseBase) -> Dict[str, Any]:
    print("Sanity Checking Graph Database...")

    visualize_graph(database)

    return {"generation": None}
