import itertools
import logging
import re
from typing import List, Optional, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.entrypoints.generate.beam_search.utils import BeamSearchInstance, create_sort_beams_key_function
from vllm.inputs import TokensPrompt

from .config import ModelConfig
from .schemas import Response, Request, Reference
from .tag_utils import preprocess_tags, postprocess_tags
from .normalization import normalize
from .prompts import PromptParts, build_prompt, prepare_references, prompt_text, reference_example
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
        if model_config.selection == 'mbr':
            from sacrebleu.metrics import CHRF
            self.chrf = CHRF(word_order=2)  # chrF++
        logger.info(f"Model loaded (translation_unit={model_config.translation_unit}, "
                    f"beam_size={model_config.beam_size}, length_penalty={model_config.length_penalty}, "
                    f"loop_guard={model_config.loop_guard}, selection={model_config.selection})")

    def _max_tokens(self, text_ids: List[int]) -> int:
        return min(self.model_config.max_new_tokens, 3 * len(text_ids) + 32)

    def _encode_prompt(self, parts: PromptParts) -> List[int]:
        """
        Tokenize the prompt as one string, as in training. If the request text (the input or the references) contains
        special token strings, the prompt is tokenized in parts and the request text with special tokens split, so that
        it cannot e.g. end the user turn.
        """
        if not any(is_request_text and any(token in part for token in self.special_tokens)
                   for part, is_request_text in parts):
            return self.tokenizer(prompt_text(parts))["input_ids"]
        input_ids = []
        for i, (is_request_text, group) in enumerate(itertools.groupby(parts, key=lambda part: part[1])):
            # only the start of the prompt gets the BOS token
            input_ids += self.tokenizer(prompt_text(list(group)), add_special_tokens=i == 0,
                                        split_special_tokens=is_request_text)["input_ids"]
        return input_ids

    def _example_lengths(self, references: Optional[List[Reference]], src_lang: str, tgt_lang: str) -> List[int]:
        """The approximate number of tokens that each reference adds to the prompt."""
        return [len(self.tokenizer(prompt_text(reference_example(self.model_config, number, reference, src_lang,
                                                                  tgt_lang)),
                                   add_special_tokens=False)["input_ids"])
                for number, reference in enumerate(references or [], 1)]

    def _fit_prompt(self, text: str, src_lang: str, tgt_lang: str, references: Optional[List[Reference]],
                    example_lengths: List[int], max_tokens: int) -> List[int]:
        """
        Encode the prompt with as many of the references as fit into max_model_len together with max_tokens of output.
        The last references are left out first; without any, the prompt without references is used.
        """
        budget = self.model_config.max_model_len - max_tokens
        count = len(references or [])
        while True:
            prompt = self._encode_prompt(build_prompt(self.model_config, text, src_lang, tgt_lang,
                                                      references[:count] if count else None))
            excess = len(prompt) - budget
            if excess <= 0 or count == 0:
                break
            # leave out references that add up to the excess, then check again as the lengths are approximate
            while count > 0 and excess > 0:
                count -= 1
                excess -= example_lengths[count]
        if references and count < len(references):
            logger.warning(f"Only {count} of {len(references)} references fit into the prompt of a segment")
        return prompt

    def _source_repetition(self, text: str, text_ids: List[int]) -> Tuple[int, bool]:
        """
        How much the source itself repeats, so that a translation may repeat as much: the most back-to-back copies of
        any block of up to loop_max_period tokens (1 if none), and whether a letter is stretched (e.g. "noooooo").
        """
        copies, n = 1, len(text_ids)
        for period in range(1, min(self.model_config.loop_max_period, n // 2) + 1):
            run = 0
            for i in range(period, n):
                run = run + 1 if text_ids[i] == text_ids[i - period] else 0
                copies = max(copies, 1 + run // period)
        return copies, re.search(r'(\w)\1\1', text) is not None

    def _find_loop(self, generated: List[int], source_copies: int = 1, stretched: bool = False) -> Optional[int]:
        """
        If the generated tokens end in one block repeated back-to-back (a decoding loop), return the length of the
        output up to and including the block's first copy, else None. A block of 2..loop_max_period tokens counts as a
        loop at loop_min_copies copies spanning at least loop_min_span tokens; a single token at loop_max_run in a row.
        Only contiguous repetition counts, so legitimate repeats (names, numbers, words) elsewhere in the output do not.
        Repetition in the source is allowed for (e.g. "whoa, whoa, whoa, whoa" -> "oi, oi, oi, oi, oi, oi"): a loop
        needs at least 2x the source's copies, a single-token run 4x, or 32 tokens if the source stretches a letter
        (stretched letters often become one-letter tokens, e.g. "noooooo" -> "неееееееееееет").
        """
        cfg, n = self.model_config, len(generated)
        min_copies = max(cfg.loop_min_copies, 2 * source_copies)
        max_run = max(cfg.loop_max_run, 4 * source_copies, 32 if stretched else 0)
        single_run = 0  # the trailing run of one token, which only the single-token limit applies to
        for period in range(1, min(cfg.loop_max_period, n // 2) + 1):
            run = 0  # trailing tokens equal to the token one period before them
            while run < n - period and generated[n - 1 - run] == generated[n - 1 - run - period]:
                run += 1
            span = run + period
            if period == 1:
                single_run = span
                if span >= max_run:
                    return n - span + 1
            elif span > single_run and span >= max(min_copies * period, cfg.loop_min_span):
                return n - span + period
        return None

    def _prune_loops(self, instance: BeamSearchInstance, prompt_length: int, seen: int, score, fallback: dict,
                     source_repetition: Tuple[int, bool]) -> int:
        """
        Remove looping beams from the live and the newly finished beams (from index `seen` on), so that the search
        continues with the others. The best looping beam, cut after the loop's first copy, is kept in `fallback`.
        Returns the number of removed beams.
        """
        removed = 0
        def keep(seq, finished):
            nonlocal removed
            generated = seq.tokens[prompt_length:]
            cut = self._find_loop(generated[:-1] if finished else generated, *source_repetition)
            if cut is None:
                return True
            removed += 1
            if "score" not in fallback or score(seq) > fallback["score"]:
                fallback.update(score=score(seq), tokens=generated[:cut])
            return False
        instance.beams = [seq for seq in instance.beams if keep(seq, False)]
        instance.completed = instance.completed[:seen] + [seq for seq in instance.completed[seen:] if keep(seq, True)]
        return removed

    def _decided(self, instance: BeamSearchInstance, score) -> bool:
        """
        Stop when there are beam_size finished translations and the worst of them scores at least as well as the
        best live beam does now (Hugging Face transformers' early_stopping=False). Without length normalization this is
        exact: a beam's score (its cumulative logprob) can only decrease as it grows.
        """
        if not instance.beams:
            return True
        beam_width = self.model_config.beam_size
        if len(instance.completed) < beam_width:
            return False
        worst_finished = sorted((score(seq) for seq in instance.completed), reverse=True)[beam_width - 1]
        return worst_finished >= max(score(seq) for seq in instance.beams)

    def _beam_search(self, prompts: List[List[int]], max_tokens: List[int],
                     source_repetition: List[Tuple[int, bool]]) -> List[List[int]]:
        """
        Beam search with vLLM's own search step, but each segment stops as soon as its result is decided, and at its
        own token limit. vLLM's LLM.beam_search runs every beam of the batch to the one max_tokens (live beams never run
        out, so it does not stop early), which made beam search >250x slower than greedy decoding.
        With loop_guard, beams that fall into a repetition loop are dropped after each step (see _find_loop).
        """
        beam_width = self.model_config.beam_size
        # 2 * beam_width candidates per step, as in vLLM's LLM.beam_search
        step_params = SamplingParams(logprobs=2 * beam_width, max_tokens=1, temperature=0.0,
                                     detokenize=False, skip_clone=True)
        # logprob / length ** length_penalty, the length counting the generated tokens with EOS; the live beams of a
        # segment all have the same length, so it only matters for comparing finished translations
        sort_key = create_sort_beams_key_function(self.model_config.length_penalty)
        engine_inputs = self.model._preprocess_cmpl([TokensPrompt(prompt_token_ids=ids) for ids in prompts])
        instances = [BeamSearchInstance(engine_input) for engine_input in engine_inputs]

        fallbacks = [{} for _ in instances]  # the best looping beam of each segment, cut after the loop's first copy
        loops = 0
        active = list(range(len(instances)))
        for step in itertools.count():
            active = [i for i in active if step < max_tokens[i] and not self._decided(instances[i], sort_key)]
            if not active:
                break
            seen = [len(instances[i].completed) for i in active]
            self.model._beam_search_step(
                instances_batch=[instances[i] for i in active],
                base_sampling_params=step_params,
                eos_token_ids=self.stop_token_ids,
                ignore_eos=False,
                beam_width=beam_width,
                sort_beams_key=sort_key,
                structured_output_backend=None,
                structured_output_key=None,
                structured_output_bitmask=None,
            )
            if self.model_config.loop_guard:
                loops += sum(self._prune_loops(instances[i], len(prompts[i]), n, sort_key, fallbacks[i],
                                               source_repetition[i])
                             for i, n in zip(active, seen))
        if loops:
            logger.info(f"Loop guard removed {loops} looping beams in "
                        f"{sum(bool(f) for f in fallbacks)}/{len(instances)} segments")
        return [self._best_beam(instance, len(ids), sort_key, fallback)
                for instance, ids, fallback in zip(instances, prompts, fallbacks)]

    def _best_beam(self, instance: BeamSearchInstance, prompt_length: int, score, fallback: dict) -> List[int]:
        """
        The best of the beam_size best finished beams, by model score or by MBR (see _mbr); if none finished, the best
        beam cut off by the token limit; if the loop guard removed every beam, the best looping one cut after its loop.
        """
        finished = [seq for seq in instance.completed
                    if len(seq.tokens) > prompt_length and seq.tokens[-1] in self.stop_token_ids]
        if finished:
            nbest = sorted(finished, key=score, reverse=True)[:self.model_config.beam_size]
            if self.model_config.selection == 'mbr' and len(nbest) > 1:
                return self._mbr([seq.tokens[prompt_length:] for seq in nbest])
            return nbest[0].tokens[prompt_length:]
        if instance.beams:
            return max(instance.beams, key=score).tokens[prompt_length:]
        return fallback.get("tokens", [])

    def _mbr(self, nbest: List[List[int]]) -> List[int]:
        """
        Minimum Bayes risk selection: the candidate with the highest total chrF++ against the other candidates, i.e.
        the consensus translation; an outlier (truncated, looping or off-track) loses. Candidates are ordered by model
        score, which breaks ties.
        """
        texts = [self.tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in nbest]
        if len(set(texts)) == 1:
            return nbest[0]
        utility = [sum(self.chrf.sentence_score(hyp, [ref]).score for j, ref in enumerate(texts) if j != i)
                   for i, hyp in enumerate(texts)]
        return nbest[max(range(len(nbest)), key=lambda i: (utility[i], -i))]

    def translate(self, texts: List[str], src_lang: str, tgt_lang: str,
                  references: Optional[List[Reference]] = None) -> List[str]:
        """
        Translate a list of segments, the prompts are tokenized the same way as in training. The references, if any,
        are included in the prompt of every segment.
        """
        example_lengths = self._example_lengths(references, src_lang, tgt_lang)
        prompts, max_tokens, source_repetition = [], [], []
        for text in texts:
            text_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            max_tokens.append(self._max_tokens(text_ids))
            prompts.append(self._fit_prompt(text.strip(), src_lang, tgt_lang, references, example_lengths,
                                            max_tokens[-1]))
            if self.model_config.loop_guard:
                source_repetition.append(self._source_repetition(text, text_ids))

        if self.model_config.beam_size > 1:
            generated = self._beam_search(prompts, max_tokens, source_repetition)
        else:
            outputs = self.model.generate(
                [TokensPrompt(prompt_token_ids=ids) for ids in prompts],
                [SamplingParams(temperature=0.0, max_tokens=n) for n in max_tokens],
                use_tqdm=False,
            )
            generated = [output.outputs[0].token_ids for output in outputs]

        return [self.tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in generated]

    def translate_chunks(self, sentences: List[str], src_lang: str, tgt_lang: str,
                         references: Optional[List[Reference]] = None) -> List[str]:
        """
        Translate consecutive sentences together, one sentence per line, to give the model more context.
        If the output line count does not match, the missing lines are left empty.
        """
        # the references count towards max_input_tokens, so they make the chunks smaller
        prompt_overhead = self.model_config.prompt_overhead + sum(self._example_lengths(references, src_lang, tgt_lang))
        chunks = chunk_by_tokens(sentences, [''] * (len(sentences) + 1), self.tokenizer,
                                 max_tokens=self.model_config.max_input_tokens,
                                 prompt_overhead=prompt_overhead)
        chunks = [chunk_sentences for chunk_sentences, _ in chunks]
        outputs = self.translate(['\n'.join(chunk) for chunk in chunks], src_lang, tgt_lang, references)

        translations = []
        for chunk, output in zip(chunks, outputs):
            lines = [line.strip() for line in output.split('\n') if line.strip()]
            if len(lines) != len(chunk):
                logger.warning(f"Chunk of {len(chunk)} sentences was translated into {len(lines)} lines")
            translations.extend((lines + [''] * len(chunk))[:len(chunk)])
        return translations

    def _references(self, request: Request) -> Optional[List[Reference]]:
        """The references to translate the request with, or None to translate without references."""
        use_references = request.use_references if request.use_references is not None else bool(request.references)
        if not use_references:
            return None
        references = prepare_references(request.references or [])
        if not references:
            #TODO use_references without references means that the model should gather its own references
            logger.warning("Gathering references is not supported yet, translating without references")
            return None
        if self.model_config.reference_prompt_template is None:
            logger.warning("No reference_prompt_template is configured, translating without references")
            return None
        return references

    def process_request(self, request: Request) -> Response:
        logger.info(f"Request received: {{"
                    f"application: {request.application}, "
                    f"input type: {request.input_type}, "
                    f"src: {request.src}, "
                    f"tgt: {request.tgt}, "
                    f"domain: {request.domain}, "
                    f"use references: {request.use_references}, "
                    f"references: {len(request.references or [])}}}")

        src_lang = self.model_config.language_codes[request.src]
        tgt_lang = self.model_config.language_codes[request.tgt]
        references = self._references(request)
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
            translated = self.translate_chunks(to_translate, src_lang, tgt_lang, references)
        else:
            translated = self.translate(to_translate, src_lang, tgt_lang, references)
        translated = iter(translated)

        translations = []
        for normalized, delimiters, tags in segmented:
            sentences = [next(translated) if sentence.strip() else '' for sentence in normalized]
            retagged = postprocess_tags(sentences, tags, request.input_type)
            translations.append(''.join(itertools.chain.from_iterable(zip(delimiters, retagged))) + delimiters[-1])
            logger.debug(f"Output: {translations[-1]}")

        return Response(result=translations[0] if isinstance(request.text, str) else translations)
