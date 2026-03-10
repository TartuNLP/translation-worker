import itertools
import logging
import warnings
from typing import List

import ctranslate2
from transformers import AutoTokenizer

from .config import ModelConfig
from .schemas import Response, Request
from .tag_utils import preprocess_tags, postprocess_tags
from .normalization import normalize
from .tokenization import sentence_tokenize
from .promptops import prep_prompt, PF_SMUGRI_MT

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', '.*__floordiv__*', )


class Translator:
    model = None
    tokenizer = None
    batch_size = 8

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self._load_model()
        logger.info("Model loaded")

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_config.tokenizer_path))
        
        self.device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        compute_type = self.model_config.compute_type
        
        logger.info(f"Initializing generator on {self.device}")
        self.model = ctranslate2.Generator(
            str(self.model_config.model_path),
            device=self.device,
            compute_type=compute_type,
            inter_threads=4
        )

    def prepare_tokens(self, prompt: str) -> List[str]:
        """Convert a prompt to a list of string tokens as expected by CTranslate2."""
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        return tokens

    def translate_batch(self, sentences: List[str], src_lang: str, tgt_lang: str, max_length: int = 2048) -> List[str]:
        """Translate a batch of sentences."""
        batch_prompts = []
        for text in sentences:
            input_data = {
                "src_segm": text.strip(),
                "src_lang": src_lang,
                "task": "translate",
                "tgt_lang": tgt_lang
            }
            prompt = prep_prompt(input_data, PF_SMUGRI_MT, inference=True)
            batch_prompts.append(prompt)

        batch_tokens = [self.prepare_tokens(prompt) for prompt in batch_prompts]
        
        results = self.model.generate_batch(
            batch_tokens,
            max_batch_size=len(batch_prompts),
            max_length=max_length,
            sampling_topk=1,
            sampling_temperature=1.0,
            include_prompt_in_result=False,
            beam_size=self.model_config.beam_size,
        )
        
        translations = []
        for result in results:
            output_ids = self.tokenizer.convert_tokens_to_ids(result.sequences[0])
            translation = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            translations.append(translation)

        return translations

    def process_request(self, request: Request) -> Response:
        logger.info(f"Request received: {{"
                    f"application: {request.application}, "
                    f"input type: {request.input_type}, "
                    f"src: {request.src}, "
                    f"tgt: {request.tgt}, "
                    f"domain: {request.domain}}}")
        
        request.src = self.model_config.language_codes[request.src]
        request.tgt = self.model_config.language_codes[request.tgt]
        inputs = [request.text] if isinstance(request.text, str) else request.text

        translations = []

        for text in inputs:
            logger.debug(f"Input: {text}")
            sentences, delimiters = sentence_tokenize(text)
            detagged, tags = preprocess_tags(sentences, request.input_type)
            normalized = [normalize(sentence) for sentence in detagged]
            
            batch_translations = []
            for i in range(0, len(normalized), self.batch_size):
                batch = normalized[i:i + self.batch_size]
                if any(batch):
                    batch_results = self.translate_batch(
                        batch,
                        src_lang=request.src,
                        tgt_lang=request.tgt
                    )
                    batch_translations.extend(batch_results)
                else:
                    batch_translations.extend([''] * len(batch))

            translated = [trans if normalized[idx] != '' else '' 
                        for idx, trans in enumerate(batch_translations)]
            retagged = postprocess_tags(translated, tags, request.input_type)
            translations.append(''.join(itertools.chain.from_iterable(
                zip(delimiters, retagged))) + delimiters[-1])
            logger.debug(f"Output: {translations[-1]}")

        response = Response(result=translations[0] if isinstance(request.text, str) else translations)

        return response
