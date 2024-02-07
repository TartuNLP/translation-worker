import os
import itertools
import logging
import warnings

import torch

from .config import ModelConfig
from .schemas import Response, Request
from .tag_utils import preprocess_tags, postprocess_tags
from .normalization import normalize, postprocess_writing_system
from .tokenization import sentence_tokenize
from .universal_interface import UniversalHubInterface
from fairseq import hub_utils

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', '.*__floordiv__*', )


class Translator:
    model = None

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self._load_model()

        logger.info("All models loaded")

    def _load_model(self):
        x = hub_utils.from_pretrained(
            model_name_or_path=self.model_config.checkpoint_path,
            checkpoint_file=self.model_config.checkpoint_file,
            data_name_or_path=self.model_config.dict_dir,
            bpe='sentencepiece',
            sentencepiece_model=self.model_config.sentencepiece_path,
            fixed_dictionary=self.model_config.dict_path
        )
        logger.info(x["args"])
        self.model = UniversalHubInterface(x["args"], x["task"], x["models"])

        if torch.cuda.is_available():
            self.model.cuda()

    def process_request(self, request: Request) -> Response:
        logger.info(f"Request received: {{"
                    f"application: {request.application}, "
                    f"input type: {request.input_type}, "
                    f"src: {request.src}, "
                    f"tgt: {request.tgt}, "
                    f"domain: {request.domain}}}")
        request.src = self.model_config.language_codes[request.src]
        request.tgt = self.model_config.language_codes[request.tgt]
        inputs = [request.text] if type(request.text) == str else request.text

        translations = []

        for text in inputs:
            logger.debug(f"Input: {text}")
            sentences, delimiters = sentence_tokenize(text)
            detagged, tags = preprocess_tags(sentences, request.input_type)
            normalized = [normalize(sentence) for sentence in detagged]
            translated = [translation if normalized[idx] != '' else '' for idx, translation in enumerate(
                self.model.translate(normalized, keep_inference_langtok=False, langtoks={'main': ('src', 'tgt')}, source_lang=request.src, target_lang=request.tgt))]
            translated = [postprocess_writing_system(sentence, request.tgt) for sentence in translated]
            retagged = postprocess_tags(translated, tags, request.input_type)
            translations.append(''.join(itertools.chain.from_iterable(zip(delimiters, retagged))) + delimiters[-1])
            logger.debug(f"Output: {translations[-1]}")

        response = Response(result=translations[0] if type(request.text) == str else translations)

        return response
