from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


class CypherQuery(BaseModel):
    query: str = Field(..., description="The Cypher query string")
    description: str = Field(..., description="A brief description of what the query does")

class CypherQueryList(BaseModel):
    queries: List[CypherQuery] = Field(..., description="List of Cypher queries")


class NodeDetectionModelOutput(BaseModel):
    node_id: Optional[str] = Field(None, description="Node id information which related with question")
    is_relevant: Optional[bool] = Field(None, description="Is the node relevant to the question")

class GenerationModelOutput(BaseModel):
    answer: str = Field(description="Answer to the question")

class AnswerQuestionModelOutput(BaseModel):
    answer: str = Field(description="Answer to the question")
    success: bool = Field(description="If model sure about the answer")

class OperationType(str, Enum):
    GENERATE_KNOWLEDGE_GRAPH = "generate_knowledge_graph"
    ANSWER_QUESTION = "answer_question"


class RouterModelOutput(BaseModel):
    operation_type: OperationType = Field(
        description="Type of operation to be performed"
    )

@dataclass
class Answer:
    answer_start: int
    text: str


@dataclass
class QuestionAnswer:
    answers: List[Answer]
    question: str
    id: str


@dataclass
class Paragraph:
    context: str
    qas: List[QuestionAnswer]


@dataclass
class Article:
    title: str
    paragraphs: List[Paragraph]
