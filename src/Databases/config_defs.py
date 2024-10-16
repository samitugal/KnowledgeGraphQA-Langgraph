from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import omegaconf
from omegaconf import OmegaConf


@dataclass
class Neo4jConfig:
    uri: str = omegaconf.MISSING
    user: str = omegaconf.MISSING
    password: str = omegaconf.MISSING
    encrypted: bool = False
    trust: str = "TRUST_ALL_CERTIFICATES"


class GDBTag(Enum):
    NEO4J = "neo4j"


@dataclass
class GDBConfig:
    tag: GDBTag = GDBTag.NEO4J


@dataclass
class MainConfig:
    gdb: GDBConfig = field(default_factory=GDBConfig)
    neo4j: Optional[Neo4jConfig] = None

    @staticmethod
    def from_file(yaml_path: str) -> "MainConfig":
        conf = OmegaConf.structured(MainConfig)
        conf = OmegaConf.merge(conf, OmegaConf.load(yaml_path))

        return conf

    @staticmethod
    def to_file(cfg, file_path: str):
        with open(file_path, "w") as f:
            yaml_str = OmegaConf.to_yaml(cfg)
            f.write(yaml_str)


if __name__ == "__main__":
    cfg = MainConfig()
    yaml_str = OmegaConf.to_yaml(cfg)

    conf = OmegaConf.structured(MainConfig)
    conf = OmegaConf.merge(conf, OmegaConf.load("./configs/neo4j.yaml"))

    MainConfig.to_file(conf, "./configs/neo4j.yaml")
    print(conf)
