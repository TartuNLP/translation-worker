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
from .tokenization import sentence_tokenize, chunk_by_tokens
from .promptops import prep_prompt, PF_SMUGRI_MT

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', '.*__floordiv__*', )


class Translator:
    model = None
    tokenizer = None
    batch_size = 8

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        # Chunking configuration from model config
        self.max_input_tokens = getattr(model_config, 'max_input_tokens', 3000)
        self.prompt_overhead = getattr(model_config, 'prompt_overhead', 100)
        # Prompt format: 'tahetorn', 'smugri_mt'
        self.prompt_format = getattr(model_config, 'prompt_format', 'tahetorn')
        self._load_model()
        logger.info(f"Model loaded (prompt_format={self.prompt_format}, max_input_tokens={self.max_input_tokens})")

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_config.tokenizer_path))
        
        self.device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        compute_type = "auto" if self.device == "cuda" else "int8_float32"
        
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

    def _build_prompt(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Build prompt based on the configured prompt format."""
        if self.prompt_format == 'smugri_mt':
            # SMUGRI MT
            data = {
                'src_segm': text.strip(),
                'src_lang': src_lang,
                'tgt_lang': tgt_lang,
                'task': 'translate'
            }
            return prep_prompt(data, PF_SMUGRI_MT, inference=True)
        else:
            # tahetorn
            return f"<start_of_turn>user\n" + \
                   f"Translate the following {src_lang} source text to {tgt_lang}:\n" + \
                   f"{src_lang}: {text.strip()}\n" + \
                   f"{tgt_lang}:<end_of_turn>\n" + \
                   f"<start_of_turn>model\n"

    def _build_chunk_prompt(self, chunk_text: str, src_lang: str, tgt_lang: str) -> str:
        """Build prompt for multi-line chunk translation."""
        if self.prompt_format == 'smugri_mt':
            # SMUGRI MT
            data = {
                'src_segm': chunk_text,
                'src_lang': src_lang,
                'tgt_lang': tgt_lang,
                'task': 'translate'
            }
            return prep_prompt(data, PF_SMUGRI_MT, inference=True)
        else:
            # tahetorn
            return f"<start_of_turn>user\n" + \
                   f"Translate the following {src_lang} text to {tgt_lang}. Preserve line breaks:\n" + \
                   f"{src_lang}:\n{chunk_text}\n" + \
                   f"{tgt_lang}:<end_of_turn>\n" + \
                   f"<start_of_turn>model\n"

    def translate_batch(self, sentences: List[str], src_lang: str, tgt_lang: str, max_length: int = 2048) -> List[str]:
        """Translate a batch of sentences."""
        batch_prompts = []
        for text in sentences:
            prompt = self._build_prompt(text, src_lang, tgt_lang)
            batch_prompts.append(prompt)

        batch_tokens = [self.prepare_tokens(prompt) for prompt in batch_prompts]
        
        results = self.model.generate_batch(
            batch_tokens,
            max_batch_size=len(batch_prompts),
            max_length=max_length,
            sampling_topk=1,
            sampling_temperature=1.0,
            include_prompt_in_result=False
        )
        
        translations = []
        for result in results:
            output_ids = self.tokenizer.convert_tokens_to_ids(result.sequences[0])
            translation = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            translations.append(translation)

        return translations

    def translate_chunk(self, sentences: List[str], delimiters: List[str], src_lang: str, tgt_lang: str, max_length: int = 4096) -> List[str]:
        """
        Translate a chunk of multiple sentences at once, preserving structure.
        
        The chunk is joined with newlines, translated as a single unit, then split back.
        This better utilizes the model's context window.
        """
        # Filter out empty sentences but track their positions
        non_empty_indices = [i for i, s in enumerate(sentences) if s.strip()]
        non_empty_sentences = [sentences[i] for i in non_empty_indices]
        
        if not non_empty_sentences:
            return [''] * len(sentences)
        
        # Join sentences with newline separator for translation
        chunk_text = '\n'.join(non_empty_sentences)
        
        # Create prompt for the entire chunk using the configured format
        prompt = self._build_chunk_prompt(chunk_text, src_lang, tgt_lang)
        
        tokens = self.prepare_tokens(prompt)
        
        results = self.model.generate_batch(
            [tokens],
            max_batch_size=1,
            max_length=max_length,
            sampling_topk=1,
            sampling_temperature=1.0,
            include_prompt_in_result=False
        )
        
        output_ids = self.tokenizer.convert_tokens_to_ids(results[0].sequences[0])
        translation = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Split translation back into lines
        translated_lines = translation.split('\n')
        
        # Handle mismatch in line count - pad or truncate as needed
        while len(translated_lines) < len(non_empty_sentences):
            translated_lines.append('')
        translated_lines = translated_lines[:len(non_empty_sentences)]
        
        # Reconstruct full list with empty strings in original positions
        result = [''] * len(sentences)
        for idx, trans in zip(non_empty_indices, translated_lines):
            result[idx] = trans.strip()
        
        return result

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
            
            # Use token-based chunking for better context utilization
            chunks = chunk_by_tokens(
                normalized, 
                delimiters, 
                self.tokenizer,
                max_tokens=self.max_input_tokens,
                prompt_overhead=self.prompt_overhead
            )
            
            all_translated = []
            all_delimiters = []
            
            for chunk_sentences, chunk_delimiters in chunks:
                if any(s.strip() for s in chunk_sentences):
                    # Translate the entire chunk at once
                    chunk_translations = self.translate_chunk(
                        chunk_sentences,
                        chunk_delimiters,
                        src_lang=request.src,
                        tgt_lang=request.tgt
                    )
                else:
                    chunk_translations = [''] * len(chunk_sentences)
                
                all_translated.extend(chunk_translations)
                # Collect delimiters (skip the first of each chunk after the first)
                if not all_delimiters:
                    all_delimiters.extend(chunk_delimiters)
                else:
                    all_delimiters.extend(chunk_delimiters[1:])
            
            # Handle case where we need the original delimiters for reconstruction
            if len(all_delimiters) < len(all_translated) + 1:
                all_delimiters = delimiters[:len(all_translated) + 1]
            
            translated = [trans if normalized[idx] != '' else '' 
                        for idx, trans in enumerate(all_translated) if idx < len(normalized)]
            retagged = postprocess_tags(translated, tags, request.input_type)
            
            # Use original delimiters for proper reconstruction
            translations.append(''.join(itertools.chain.from_iterable(
                zip(delimiters[:len(retagged)], retagged))) + delimiters[len(retagged)] if len(delimiters) > len(retagged) else '')
            logger.debug(f"Output: {translations[-1]}")

        response = Response(result=translations[0] if isinstance(request.text, str) else translations)

        return response
