import yaml
from yaml.loader import SafeLoader
from typing import List, Dict, Optional

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
    nmt_model_path: str  # path to the sockeye model directory
    spm_model: str
    tc_model: Optional[str]
    domains: List[Domain]
    factor_sequence: List[str]
    language_codes: Dict[str, str]
    domain_codes: Dict[str, str]


def read_model_config(file_path: str, model_name: str) -> ModelConfig:
    with open(file_path, 'r', encoding='utf-8') as f:
        model_config = ModelConfig(model_name=model_name, **yaml.load(f, Loader=SafeLoader)['models'][model_name])

    return model_config
