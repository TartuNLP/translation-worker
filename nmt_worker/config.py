import yaml
from yaml.loader import SafeLoader
from typing import List, Dict, Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class MQConfig(BaseSettings):
    """
    Imports MQ configuration from environment variables
    """
    host: str = 'localhost'
    port: int = 5672
    username: str = 'guest'
    password: str = 'guest'
    exchange: str = 'translation'
    heartbeat: int = 60
    connection_name: str = 'Translation worker'

    class Config:
        env_file = 'config/.env'
        env_prefix = 'mq_'


class Domain(BaseModel):
    name: str
    language_pairs: List[str]  # a list of hyphen-separated input/output language pairs


class ModelConfig(BaseModel):
    model_name: str
    model_path: str
    tokenizer_path: str
    domains: List[Domain]
    language_codes: Dict[str, str]
    
    class Config:
        # Allow fields that start with 'model_'
        protected_namespaces = ()


def read_model_config(file_path: str) -> ModelConfig:
    with open(file_path, 'r', encoding='utf-8') as f:
        model_config = ModelConfig(**yaml.load(f, Loader=SafeLoader)['model'])

    return model_config
