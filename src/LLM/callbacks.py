import time
from typing import Any, Dict, List

from langchain.schema import LLMResult
from langchain_core.callbacks import BaseCallbackHandler


class LLMCallbacks(BaseCallbackHandler):
    def __init__(self):
        super(LLMCallbacks, self).__init__()
        self.starttime = None
        self.duration: int = 0
        self.model_name: str = None

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""

        self.model_name = self._extract_model_info(serialized)
        self.starttime = time.time()

    def _extract_model_info(self, serialized: Dict[str, Any]) -> str:
        """
        Extracts LLM Model information from kwargs if exists
        """
        model_info = (
            serialized.get("kwargs", {})
            .get("model_kwargs", {})
            .get("messages", [{}])[0]
            .get("model_info")
        )
        return model_info if model_info else serialized["name"]

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        self.duration = time.time() - self.starttime
        print(f"{self.model_name.capitalize()} Duration => {self.duration} seconds")
