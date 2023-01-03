import itertools
import logging
import warnings

from .config import ModelConfig
from .schemas import Response, Request
from .sockeye_model import SockeyeModel
from .tag_utils import preprocess_tags, postprocess_tags
from .normalization import normalize
from .tokenization import sentence_tokenize

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', '.*__floordiv__*', )


class Translator:
    model = None

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self._load_model()

        logger.info("All models loaded")

    def _load_model(self):
        self.model = SockeyeModel(
            nmt_model_path=self.model_config.nmt_model_path,
            spm_model=self.model_config.spm_model,
            tc_model=self.model_config.tc_model,
            factor_sequence=self.model_config.factor_sequence
        )

    def process_request(self, request: Request) -> Response:
        logger.info(f"Request received: {{"
                    f"application: {request.application}, "
                    f"input type: {request.input_type}, "
                    f"domain: {request.domain}, "
                    f"tgt: {request.tgt}}}")
        sent_factors = {'lang': self.model_config.language_codes[request.tgt]}
        if 'domain' in self.model.factor_sequence:
            if request.domain is not None and request.domain in self.model_config.domain_codes:
                sent_factors['domain'] = self.model_config.domain_codes[request.domain]
            else:
                sent_factors['domain'] = list(self.model_config.domain_codes.keys())[0]

        inputs = [request.text] if type(request.text) == str else request.text

        translations = []

        for text in inputs:
            sentences, delimiters = sentence_tokenize(text)
            detagged, tags = preprocess_tags(sentences, request.input_type)
            normalized = [normalize(sentence) for sentence in detagged]
            translated, _, _, _ = self.model.translate(normalized, sent_factors)
            retagged = postprocess_tags(translated, tags, request.input_type)
            translations.append(''.join(itertools.chain.from_iterable(zip(delimiters, retagged))) + delimiters[-1])

        response = Response(result=translations[0] if type(request.text) == str else translations)

        return response
