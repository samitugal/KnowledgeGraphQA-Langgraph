import string
from typing import Any, Dict, List, TypeVar

from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel

from ..output_models import *
from ..Utils.prompt_renderer import load_prompt
from .LLMAbstractBase import LLMAbstractBase

U = TypeVar("U", bound=BaseModel)


class LLMBase(LLMAbstractBase):
    def __init__(self, config):
        self.config = config
        load_dotenv()

    def generate_kg_query(self, content: str) -> CypherQueryList:
        """
        This function generates a knowledge graph query based on the provided content.
        It uses a pre-defined template for the knowledge graph generation and a structured output parser.
        The function returns the generated knowledge graph query.
        """
        knowledge_graph_template = load_prompt("generate_knowledge_graph")
        output_parser = PydanticOutputParser(pydantic_object=CypherQueryList)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", knowledge_graph_template),
            ("human", f"Can you generate a knowledge graph about my {content}?"),
            ("human", "Format the output as follows:\n{format_instructions}")
        ]).partial(format_instructions=output_parser.get_format_instructions())

        chain = prompt | self.client | output_parser
        response = chain.invoke({"content": content})
        print(response)
        return response

    def router_model(self, user_request: str) -> RouterModelOutput:
        """
        This function acts as a router model that determines the appropriate model for processing the user's request.
        It uses a pre-defined template for the router model and a structured output parser.
        The function returns the appropriate model for processing the user's request.
        """
        router_model_template = load_prompt("router_model")
        output_parser = PydanticOutputParser(pydantic_object=RouterModelOutput)
        router_model_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", router_model_template),
                ("human", "{user_request}"),
                ("human", "Format the output as follows:\n{format_instructions}")
            ]
        ).partial(format_instructions=output_parser.get_format_instructions())

        chain = router_model_prompt | self.client | output_parser
        response = chain.invoke({"user_request": user_request})
        return response

    def detect_target_node(
        self, content: str, graphdb_nodes: List[Dict[str, Any]]
    ) -> NodeDetectionModelOutput:
        """
        This function detects the target node for the given content and graph database nodes.
        It uses a pre-defined template for the target node detection and a structured output parser.
        The function returns the detected target node.
        """
        target_node_template = load_prompt("detect_target_node")
        structured_llm_generator = self.client.with_structured_output(
            NodeDetectionModelOutput
        )
        target_node_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", target_node_template),
                (
                    "human",
                    "The content is {content} \\n\\n The graphs are {graphdb_nodes}",
                ),
            ]
        )
        chain = target_node_prompt | structured_llm_generator
        response = chain.invoke({"content": content, "graphdb_nodes": graphdb_nodes})
        return response
