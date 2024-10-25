import os
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from pydantic_settings import BaseSettings
from langfuse.callback import CallbackHandler
from langchain_core.messages import HumanMessage

from src.const import *
from src.ContentProvider.content_provider import ContentProvider
from src.Databases.config_defs import MainConfig
from src.Databases.GraphDatabase import GraphDatabase
from src.LLM.config_defs import LLMMainConfig
from src.LLM.Pipeline import Pipeline
from src.Nodes import *
from src.output_models import OperationType, RouterModelOutput
from src.state_model import GraphState

load_dotenv()


class MainRuntimeVars(BaseSettings):
    GRAPH_DATABASE_CONNECTION_PATH: str
    LLM_CONFIG_PATH: str

    class Config:
        env_file = ".env"
        extra = "allow"


envvars = MainRuntimeVars()

database_config: MainConfig = MainConfig.from_file(
    envvars.GRAPH_DATABASE_CONNECTION_PATH
)
print(f"Using config from {envvars.GRAPH_DATABASE_CONNECTION_PATH}")
llm_config: LLMMainConfig = LLMMainConfig.from_file(envvars.LLM_CONFIG_PATH)
print(f"Using config from {envvars.LLM_CONFIG_PATH}")

content = ContentProvider()
llm = Pipeline.new_instance_from_config(config=llm_config)
database = GraphDatabase.new_instance_from_config(config=database_config)

langfuse_handler = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", os.getenv("LANGFUSE_HOST_LOCAL"))
)


def decide_to_generate(state):
    print("Deciding to generate...")
    router_output = state['operation_type']
    if router_output.operation_type == OperationType.ANSWER_QUESTION:
        print("-- Routing to Answer Question")
        return FIND_TARGET_NODE
    elif router_output.operation_type == OperationType.GENERATE_KNOWLEDGE_GRAPH:
        print("-- Routing to Knowledge Graph Generator")
        return KNOWLEDGE_GRAPH_GENERATOR_NODE
    else:
        print(f"-- Decision: Unknown operation type: {router_output.operation_type}")
        return END
    
def is_relevant(state):
    is_relevant = state["is_relevant"]
    related_node_id = state["result"]

    if is_relevant:
        print(f"-- Decision: Node is relevant: {related_node_id}")
        return ANSWER_QUESTION_NODE
    else:
        print("-- Decision: Node is not relevant")
        return WEB_SEARCH_NODE

def graph_sanity_check(state):
    result = state["generation"]
    if result:
        print("-- Sanity Check: Graph is valid")
        return VISUALIZE_GRAPH_NODE
    else:
        print("-- Sanity Check: Graph is invalid")
        return KNOWLEDGE_GRAPH_GENERATOR_NODE

workflow = StateGraph(GraphState)

workflow.add_node(ROUTER_NODE, lambda state: router_node(state, llm))
workflow.add_node(KNOWLEDGE_GRAPH_GENERATOR_NODE, lambda state: knowledge_graph_generator_node(state, llm))
workflow.add_node(ANSWER_QUESTION_NODE, lambda state: answer_question_node(state, llm, database))
workflow.add_node(EXECUTE_CYPHER_QUERY_NODE, lambda state: execute_cypher_query_node(state, database))
workflow.add_node(GRAPH_SANITY_CHECK_NODE, lambda state: graph_db_sanity_check_node(state, database))
workflow.add_node(VISUALIZE_GRAPH_NODE, lambda state: visualize_graph_node(state, database))
workflow.add_node(FIND_TARGET_NODE, lambda state: find_target_node(state, llm, database))
workflow.add_node(WEB_SEARCH_NODE, web_search_node)
workflow.add_node(GENERATION_NODE, lambda state: generation_node(state, llm))

workflow.set_entry_point(ROUTER_NODE)
workflow.add_conditional_edges(
    ROUTER_NODE,
    decide_to_generate,
    {
        FIND_TARGET_NODE: FIND_TARGET_NODE,
        KNOWLEDGE_GRAPH_GENERATOR_NODE: KNOWLEDGE_GRAPH_GENERATOR_NODE,
    },
)
workflow.add_conditional_edges(
    GRAPH_SANITY_CHECK_NODE,
    graph_sanity_check,
    {
        KNOWLEDGE_GRAPH_GENERATOR_NODE: KNOWLEDGE_GRAPH_GENERATOR_NODE,
        VISUALIZE_GRAPH_NODE: VISUALIZE_GRAPH_NODE,
    },
)
workflow.add_edge(KNOWLEDGE_GRAPH_GENERATOR_NODE, EXECUTE_CYPHER_QUERY_NODE)
workflow.add_edge(EXECUTE_CYPHER_QUERY_NODE, GRAPH_SANITY_CHECK_NODE)
workflow.add_edge(VISUALIZE_GRAPH_NODE, END)

workflow.add_conditional_edges(
    FIND_TARGET_NODE,
    is_relevant,
    {
        ANSWER_QUESTION_NODE: ANSWER_QUESTION_NODE,
        WEB_SEARCH_NODE: WEB_SEARCH_NODE,
    },
)
workflow.add_edge(WEB_SEARCH_NODE, GENERATION_NODE)
workflow.add_edge(ANSWER_QUESTION_NODE, GENERATION_NODE)
workflow.add_edge(GENERATION_NODE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph-schema.png")
