import warnings
from typing import Any, Dict, List, TypeAlias
from dotenv import load_dotenv
from neo4j import GraphDatabase

from .config_defs import MainConfig
from .DatabaseBase import DatabaseBase

warnings.filterwarnings("ignore", category=DeprecationWarning)

QueryOutputDict: TypeAlias = List[Dict[str, Any]]


class Neo4jDatabase(DatabaseBase):
    instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self, config: MainConfig):
        if not hasattr(self, "initialized"):  # Singleton kontrolü
            super().__init__(config)
            load_dotenv()
            AUTH = (config.neo4j.user, config.neo4j.password)
            # Driver'ı burada kalıcı olarak oluşturuyoruz.
            self.driver = GraphDatabase.driver(config.neo4j.uri, auth=AUTH)
            self.driver.verify_connectivity()
            self.initialized = True

    def sanity_check(self) -> bool:
        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as node_count")
            record = result.single()  # Tek sonuç döndüğünden single() kullanıyoruz
            return record["node_count"] > 0

    def fetch_all(self) -> QueryOutputDict:
        """Executes Query for Neo4j"""
        with self.driver.session() as session:
            result = session.write_transaction(self._fetch_all)
            return list(result)  # Result verisini işlem içinde listeye çeviriyoruz

    @staticmethod
    def _fetch_all(tx):
        result = tx.run("MATCH (n)-[r]->(m) RETURN n, r, m")
        return result.values()  # Doğrudan listeye dönüştür

    def execute_cypher_query(self, query: str):
        """Executes Query for Neo4j"""
        with self.driver.session() as session:
            session.write_transaction(self._execute_cypher_query, query)

    @staticmethod
    def _execute_cypher_query(tx, query: str):
        return tx.run(query)

    def flush_all(self):
        with self.driver.session() as session:
            session.write_transaction(self._flush_all)

    @staticmethod
    def _flush_all(tx):
        tx.run("MATCH (n) DETACH DELETE n")

    def list_nodes_and_properties(self) -> QueryOutputDict:
        nodes_list = []
        with self.driver.session() as session:
            result = session.run(
                "MATCH (n) RETURN n, labels(n) AS labels, keys(n) AS keys, properties(n) AS properties"
            )
            for record in result:
                node = {
                    "id": record["n"].element_id,
                    "labels": record["labels"],
                    "keys": record["keys"],
                    "properties": record["properties"],
                }
                nodes_list.append(node)
        return nodes_list

    def list_n_degree_nodes(self, node_id: str, degree_count: int) -> QueryOutputDict:
        nodes_list = []
        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (n)-[r*1..{degree_count}]-(m)
                WHERE elementId(n) = "{node_id}"
                RETURN m, r
                """
            )
            for record in result:
                node = record["m"]
                relationships = record["r"]

                node_data = {
                    "id": node.element_id,
                    "labels": list(node.labels),
                    "properties": dict(node),
                    "relationships": [],
                }

                for relationship in relationships:
                    relationship_data = {
                        "type": relationship.type,
                        "properties": dict(relationship),
                        "start_node": relationship.start_node.element_id,
                        "end_node": relationship.end_node.element_id,
                    }
                    node_data["relationships"].append(relationship_data)

                nodes_list.append(node_data)

        return nodes_list

    def disconnect(self) -> None:
        if self.driver:
            self.driver.close()


if __name__ == "__main__":
    config = MainConfig.from_file("configs/GraphDatabase/neo4j.yaml")
    print(config)
    db = Neo4jDatabase(config)
    print(db.sanity_check())
