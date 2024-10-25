import os
import string
from typing import Any, Dict, List, TypeVar

from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import StructuredOutputParser
from pydantic import BaseModel
from langfuse.decorators import langfuse_context, observe
from langfuse.callback import CallbackHandler

from ..output_models import *
from ..Utils.prompt_renderer import load_prompt
from .LLMAbstractBase import LLMAbstractBase

U = TypeVar("U", bound=BaseModel)


class LLMBase(LLMAbstractBase):
    def __init__(self, config):
        self.config = config
        load_dotenv()
        self.langfuse_handler = CallbackHandler(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", os.getenv("LANGFUSE_HOST_LOCAL"))
        )

    @observe(name="knowledge_graph_generation")
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
        
        chain = (prompt | self.client | output_parser).with_config(
            callbacks=[self.langfuse_handler],
            metadata={"run_name": "Knowledge Graph Generator"},
            run_name="Knowledge Graph Generator"
        )
        response = chain.invoke({"content": content})
        return response
    

    @observe(name="router_model_execution")
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
        response = chain.invoke(
            {"user_request": user_request}, 
            {"callbacks": [self.langfuse_handler], "metadata": {"run_name": "Router Model"}, "run_name": "Router Model"}
        )
        return response

    @observe(name="detect_target_node")
    def detect_target_node(
        self, content: str, graphdb_nodes: List[Dict[str, Any]]
    ) -> NodeDetectionModelOutput:
        """
        This function detects the target node for the given content and graph database nodes.
        It uses a pre-defined template for the target node detection and a structured output parser.
        The function returns the detected target node.
        """
        target_node_template = load_prompt("detect_target_node")
        output_parser = PydanticOutputParser(pydantic_object=NodeDetectionModelOutput)
        target_node_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", target_node_template),
                ("human", "The content is {content}"),
                ("human", "The graphs are {graphdb_nodes}"),
                ("human", "Format the output as follows:\n{format_instructions}")
            ]
        ).partial(format_instructions=output_parser.get_format_instructions())
        chain = target_node_prompt | self.client | output_parser
        response = chain.invoke(
            {"content": content, "graphdb_nodes": str(graphdb_nodes)},
            {"callbacks": [self.langfuse_handler], "metadata": {"run_name": "Target Node Detection"}, "run_name": "Target Node Detection"}
        )
        return response
    
    @observe(name="answer_question_model")
    def answer_question_model(self, context: str, question: str) -> AnswerQuestionModelOutput:
        answer_question_model_template = load_prompt("answer_question_model")
        output_parser = PydanticOutputParser(pydantic_object=AnswerQuestionModelOutput)
        answer_question_model_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", answer_question_model_template),
                ("human", "{context}"),
                ("human", "{question}"),
                ("human", "Format the output as follows:\n{format_instructions}")
            ]
        ).partial(format_instructions=output_parser.get_format_instructions(), 
                  config={"callbacks": [self.langfuse_handler], "metadata": {"run_name": "Answer Question Model"}, "run_name": "Answer Question Model"})

        chain = answer_question_model_prompt | self.client | output_parser
        response = chain.invoke({"context": context, "question": question})
        return response

    @observe(name="generation_model")
    def generation_model(self, context: str, question: str) -> GenerationModelOutput:
        generation_model_template = load_prompt("generation_model")
        output_parser = PydanticOutputParser(pydantic_object=GenerationModelOutput)
        generation_model_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", generation_model_template),
                ("human", "{context}"),
                ("human", "{question}"),
                ("human", "Format the output as follows:\n{format_instructions}")
            ]
        ).partial(format_instructions=output_parser.get_format_instructions())

        chain = generation_model_prompt | self.client | output_parser
        response = chain.invoke({"context": context, "question": question}, config={"callbacks": [self.langfuse_handler], 
                                                                                    "metadata": {"run_name": "Generation Model"}, "run_name": "Generation Model"})
        return response
