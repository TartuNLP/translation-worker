import os
import itertools
import logging
import warnings

import torch

from .config import ModelConfig
from .schemas import Response, Request
from .tag_utils import preprocess_tags, postprocess_tags
from .normalization import normalize
from .tokenization import sentence_tokenize
from .modular_interface import ModularHubInterface

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', '.*__floordiv__*', )


class Translator:
    model = None

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self._load_model()

        logger.info("All models loaded")

    def _load_model(self):
        sentencepiece_path = os.path.join(self.model_config.sentencepiece_dir, self.model_config.sentencepiece_prefix)
        self.model = ModularHubInterface.from_pretrained(
            model_path=self.model_config.checkpoint_path,
            sentencepiece_prefix=sentencepiece_path,
            dictionary_path=self.model_config.dict_dir)
        if torch.cuda.is_available():
            self.model.cuda()

    def process_request(self, request: Request) -> Response:
        logger.info(f"Request received: {{"
                    f"application: {request.application}, "
                    f"input type: {request.input_type}, "
                    f"src: {request.src}, "
                    f"tgt: {request.tgt}, "
                    f"domain: {request.domain}}}")
        request.true_src = self.model_config.internal_language_codes[request.src]
        request.src = self.model_config.language_codes[request.src]
        request.tgt = self.model_config.language_codes[request.tgt]
        inputs = [request.text] if type(request.text) == str else request.text

        translations = []

        for text in inputs:
            sentences, delimiters = sentence_tokenize(text)
            detagged, tags = preprocess_tags(sentences, request.input_type)
            normalized = [normalize(sentence) for sentence in detagged]
            translated = [translation if normalized[idx] != '' else '' for idx, translation in enumerate(
                self.model.translate(normalized, src_language=request.src, tgt_language=request.tgt,
                                     true_src_language=request.true_src)
            )]
            retagged = postprocess_tags(translated, tags, request.input_type)
            translations.append(''.join(itertools.chain.from_iterable(zip(delimiters, retagged))) + delimiters[-1])

        response = Response(result=translations[0] if type(request.text) == str else translations)

        return response
