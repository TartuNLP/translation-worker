import yaml
from yaml.loader import SafeLoader
from typing import List, Dict

from pydantic import BaseSettings, BaseModel


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
    checkpoint_path: str
    dict_dir: str
    sentencepiece_dir: str
    sentencepiece_prefix: str
    domains: List[Domain]
    language_codes: Dict[str, str]
    internal_language_codes: Dict[str, str]


def read_model_config(file_path: str, model_name: str) -> ModelConfig:
    with open(file_path, 'r', encoding='utf-8') as f:
        model_config = ModelConfig(model_name=model_name, **yaml.load(f, Loader=SafeLoader)['models'][model_name])

    return model_config
