import yaml
from yaml.loader import SafeLoader
from typing import List, Dict, Optional, Literal

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
    tokenizer_path: Optional[str] = None  # defaults to model_path
    domains: List[Domain]
    language_codes: Dict[str, str]
    # Prompt with {src_lang}, {tgt_lang} and {text} placeholders
    prompt_template: str
    # 'sentence' or 'chunk' (consecutive sentences up to max_input_tokens, one sentence per line)
    translation_unit: Literal['sentence', 'chunk'] = 'sentence'
    max_input_tokens: int = 1500
    prompt_overhead: int = 100
    beam_size: int = 1
    length_penalty: float = 1.0
    # Beam search only: 'score' returns the best finished translation by model score, 'mbr' the consensus of the
    # beam_size best ones (highest chrF++ against the others)
    selection: Literal['score', 'mbr'] = 'score'
    # Beam search only: drop beams that end in a block of tokens repeated back-to-back (see Translator._find_loop)
    loop_guard: bool = False
    loop_max_period: int = 40  # longest repeated block checked, in tokens
    loop_min_copies: int = 3   # a block of 2+ tokens is a loop at this many back-to-back copies...
    loop_min_span: int = 16    # ...spanning at least this many tokens
    loop_max_run: int = 8      # a single token is a loop at this many in a row
    max_new_tokens: int = 1024
    # vLLM engine
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.85
    max_num_seqs: int = 64

    class Config:
        # Allow fields that start with 'model_'
        protected_namespaces = ()


def read_model_config(file_path: str) -> ModelConfig:
    with open(file_path, 'r', encoding='utf-8') as f:
        model_config = ModelConfig(**yaml.load(f, Loader=SafeLoader)['model'])

    return model_config
