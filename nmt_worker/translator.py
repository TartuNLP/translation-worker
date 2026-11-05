import itertools
import logging
from typing import List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.sampling_params import BeamSearchParams

from .config import ModelConfig
from .schemas import Response, Request
from .tag_utils import preprocess_tags, postprocess_tags
from .normalization import normalize
from .tokenization import sentence_tokenize, chunk_by_tokens

logger = logging.getLogger(__name__)


class Translator:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        tokenizer_path = model_config.tokenizer_path or model_config.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = LLM(
            model=model_config.model_path,
            tokenizer=tokenizer_path,
            max_model_len=model_config.max_model_len,
            gpu_memory_utilization=model_config.gpu_memory_utilization,
            max_num_seqs=model_config.max_num_seqs,
            seed=0,
        )
        # strings of special tokens (e.g. <|assistant_end|>) that must not be recognized in user text
        self.special_tokens = [token.content for token in self.tokenizer.added_tokens_decoder.values() if token.special]
        # beam search stops at these tokens (tokenizer EOS + generation config EOS, as in vLLM)
        eos = self.model.llm_engine.input_processor.generation_config_fields.get("eos_token_id")
        self.stop_token_ids = {self.tokenizer.eos_token_id, *(eos if isinstance(eos, list) else [eos] if eos else [])}
        logger.info(f"Model loaded (translation_unit={model_config.translation_unit}, "
                    f"beam_size={model_config.beam_size})")

    def _max_tokens(self, text_ids: List[int]) -> int:
        return min(self.model_config.max_new_tokens, 3 * len(text_ids) + 32)

    def _encode_prompt(self, text: str, src_lang: str, tgt_lang: str) -> List[int]:
        """
        Tokenize the prompt as one string, as in training. If the user text contains special token strings, the text
        is tokenized separately with special tokens split, so that it cannot e.g. end the user turn.
        """
        template = self.model_config.prompt_template
        if not any(token in text for token in self.special_tokens):
            return self.tokenizer(template.format(src_lang=src_lang, tgt_lang=tgt_lang, text=text))["input_ids"]
        prefix, suffix = (part.format(src_lang=src_lang, tgt_lang=tgt_lang) for part in template.split("{text}", 1))
        return (self.tokenizer(prefix)["input_ids"]
                + self.tokenizer(text, add_special_tokens=False, split_special_tokens=True)["input_ids"]
                + self.tokenizer(suffix, add_special_tokens=False)["input_ids"])

    def _best_beam(self, output, prompt_length: int) -> List[int]:
        """
        vLLM also returns beams that were cut off by max_tokens, and with length normalization a long unfinished
        beam can outscore the finished ones (sequences are sorted by score). Prefer the best finished beam.
        """
        finished = [seq for seq in output.sequences
                    if len(seq.tokens) > prompt_length and seq.tokens[-1] in self.stop_token_ids]
        return (finished or output.sequences)[0].tokens[prompt_length:]

    def translate(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        """Translate a list of segments, the prompts are tokenized the same way as in training."""
        prompts, max_tokens = [], []
        for text in texts:
            prompts.append(self._encode_prompt(text.strip(), src_lang, tgt_lang))
            max_tokens.append(self._max_tokens(self.tokenizer(text, add_special_tokens=False)["input_ids"]))

        if self.model_config.beam_size > 1:
            outputs = self.model.beam_search(
                [TokensPrompt(prompt_token_ids=ids) for ids in prompts],
                # vLLM runs all beams up to max_tokens and then ranks the finished and the unfinished beams together;
                # with length normalization, long degenerate unfinished beams outrank the finished translations
                BeamSearchParams(beam_width=self.model_config.beam_size, max_tokens=max(max_tokens), length_penalty=0.0),
            )
            # beam search returns the prompt and the output together
            generated = [self._best_beam(output, len(ids)) for output, ids in zip(outputs, prompts)]
        else:
            outputs = self.model.generate(
                [TokensPrompt(prompt_token_ids=ids) for ids in prompts],
                [SamplingParams(temperature=0.0, max_tokens=n) for n in max_tokens],
                use_tqdm=False,
            )
            generated = [output.outputs[0].token_ids for output in outputs]

        return [self.tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in generated]

    def translate_chunks(self, sentences: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        """
        Translate consecutive sentences together, one sentence per line, to give the model more context.
        If the output line count does not match, the missing lines are left empty.
        """
        chunks = chunk_by_tokens(sentences, [''] * (len(sentences) + 1), self.tokenizer,
                                 max_tokens=self.model_config.max_input_tokens,
                                 prompt_overhead=self.model_config.prompt_overhead)
        chunks = [chunk_sentences for chunk_sentences, _ in chunks]
        outputs = self.translate(['\n'.join(chunk) for chunk in chunks], src_lang, tgt_lang)

        translations = []
        for chunk, output in zip(chunks, outputs):
            lines = [line.strip() for line in output.split('\n') if line.strip()]
            if len(lines) != len(chunk):
                logger.warning(f"Chunk of {len(chunk)} sentences was translated into {len(lines)} lines")
            translations.extend((lines + [''] * len(chunk))[:len(chunk)])
        return translations

    def process_request(self, request: Request) -> Response:
        logger.info(f"Request received: {{"
                    f"application: {request.application}, "
                    f"input type: {request.input_type}, "
                    f"src: {request.src}, "
                    f"tgt: {request.tgt}, "
                    f"domain: {request.domain}}}")

        src_lang = self.model_config.language_codes[request.src]
        tgt_lang = self.model_config.language_codes[request.tgt]
        inputs = [request.text] if isinstance(request.text, str) else request.text

        # all sentences of all inputs are translated in one call, vLLM batches them internally
        segmented = []
        for text in inputs:
            logger.debug(f"Input: {text}")
            sentences, delimiters = sentence_tokenize(text)
            detagged, tags = preprocess_tags(sentences, request.input_type)
            normalized = [normalize(sentence) for sentence in detagged]
            segmented.append((normalized, delimiters, tags))

        to_translate = [sentence for normalized, _, _ in segmented for sentence in normalized if sentence.strip()]
        if not to_translate:
            translated = []
        elif self.model_config.translation_unit == 'chunk':
            translated = self.translate_chunks(to_translate, src_lang, tgt_lang)
        else:
            translated = self.translate(to_translate, src_lang, tgt_lang)
        translated = iter(translated)

        translations = []
        for normalized, delimiters, tags in segmented:
            sentences = [next(translated) if sentence.strip() else '' for sentence in normalized]
            retagged = postprocess_tags(sentences, tags, request.input_type)
            translations.append(''.join(itertools.chain.from_iterable(zip(delimiters, retagged))) + delimiters[-1])
            logger.debug(f"Output: {translations[-1]}")

        return Response(result=translations[0] if isinstance(request.text, str) else translations)
