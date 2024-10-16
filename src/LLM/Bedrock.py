import warnings

import boto3
from langchain_aws import ChatBedrock

from .callbacks import LLMCallbacks
from .config_defs import LLMMainConfig, LLMTag
from .LLMBase import LLMBase

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")


class Bedrock(LLMBase):
    def __init__(self, config):
        super().__init__(config)
        if config.llm.llm_tag != LLMTag.BEDROCK:
            raise ValueError("BedrockPipeline can only be used with Bedrock")
        if config.bedrock is None:
            raise ValueError("BedrockPipeline requires a BedrockConfig")

        bedrock = boto3.client(
            service_name="bedrock-runtime", region_name=config.bedrock.region_name
        )
        self.client = ChatBedrock(
            model_id=config.bedrock.model_id,
            client=bedrock,
            model_kwargs={"temperature": config.llm.temperature},
            callbacks=[LLMCallbacks()],
        )
