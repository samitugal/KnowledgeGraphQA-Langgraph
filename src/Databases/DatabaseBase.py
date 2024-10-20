from abc import ABC, abstractmethod

from .config_defs import GDBTag, MainConfig


class DatabaseBase(ABC):
    def __init__(self, config: MainConfig):
        self.config = config

    @abstractmethod
    def execute_cypher_query(self, query: str):
        pass

    @abstractmethod
    def fetch_all(self):
        pass

    @abstractmethod
    def sanity_check(self) -> bool:
        pass

    @abstractmethod
    def list_n_degree_nodes(self, node_id: str, degree_count: int):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def flush_all(self):
        pass
