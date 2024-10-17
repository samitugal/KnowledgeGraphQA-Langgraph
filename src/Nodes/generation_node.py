from typing import Any, Dict

from src.state_model import GraphState

def generation_node(state: GraphState) -> Dict[str, Any]:
    print("Generating...")

    question = state["question"]
    documents = state["documents"]

    #result = generation_chain.invoke({"context": documents, "question": question})
    #return {"documents": documents, "question": question, "generation": result}
