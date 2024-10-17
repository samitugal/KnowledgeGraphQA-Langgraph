from typing import Any, Dict

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.tools.tavily_search import *
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from src.state_model import GraphState

load_dotenv()

web_search_tool = TavilySearchResults(api_wrapper=TavilySearchAPIWrapper(), max_results=3) # type: ignore

def web_search_node(state: GraphState) -> Dict[str, Any]:
    """
    Performs web search for the given question.
    """

    question = state["question"]

    web_search_results = web_search_tool.invoke({"query": question})
    concatted_tavity_results = "\n".join([d["content"] for d in web_search_results])

    web_results = Document(page_content=concatted_tavity_results, metadata={"source": "web"})

    return {"result": web_results, "question": question}