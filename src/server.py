from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from .ContentProvider.content_provider import ContentProvider
from .Databases.config_defs import MainConfig
from .Databases.GraphDatabase import GraphDatabase
from .LLM.config_defs import LLMMainConfig
from .LLM.Pipeline import Pipeline


class MainRuntimeVars(BaseSettings):
    GRAPH_DATABASE_CONNECTION_PATH: str
    LLM_CONFIG_PATH: str

    class Config:
        env_file = ".env"
        extra = "allow"


app = FastAPI()
envvars = MainRuntimeVars()

load_dotenv()

database_config: MainConfig = MainConfig.from_file(
    envvars.GRAPH_DATABASE_CONNECTION_PATH
)
print(f"Using config from {envvars.GRAPH_DATABASE_CONNECTION_PATH}")
llm_config: LLMMainConfig = LLMMainConfig.from_file(envvars.LLM_CONFIG_PATH)
print(f"Using config from {envvars.LLM_CONFIG_PATH}")


@app.post("/generate_knowledge_graph", status_code=status.HTTP_200_OK)
def generate_response():
    try:
        content = ContentProvider()
        llm = Pipeline.new_instance_from_config(config=llm_config)
        response = llm.generate_kg_query(content.all_contexts)
        database = GraphDatabase.new_instance_from_config(config=database_config)
        ##
        database.flush_all()
        for query in response.queries:
            print(query)
            database.execute_query(query=query)

        return response

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
