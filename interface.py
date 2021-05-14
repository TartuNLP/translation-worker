import os
import copy
from typing import Dict, List, Iterator, Any, Optional

from fairseq.data import Dictionary, LanguagePairDataset, FairseqDataset
from fairseq import utils, search, hub_utils
from fairseq.models.multilingual_transformer import MultilingualTransformerModel
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
from fairseq.sequence_generator import SequenceGenerator

from omegaconf import open_dict, DictConfig

from sentencepiece import SentencePieceProcessor

import torch
from torch import Tensor, LongTensor
from torch.nn import ModuleList, Module


class ModularHubInterface(Module):
    def __init__(
            self,
            models: List[MultilingualTransformerModel],
            task: MultilingualTranslationTask,
            cfg: DictConfig,
            sp_models: Dict[str, SentencePieceProcessor]
    ):
        super().__init__()

        self.sp_models = sp_models
        self.models = ModuleList(models)
        self.task = task
        self.cfg = cfg
        self.dicts: Dict[str, Dictionary] = task.dicts
        self.langs = task.langs

        for model in self.models:
            model.prepare_for_inference_(self.cfg)

        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(), *[model.max_positions() for model in self.models]
        )

        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    @classmethod
    def from_pretrained(
            cls,
            model_path: str,
            sentencepiece_prefix: str,
            dictionary_path: str,
    ):
        x = hub_utils.from_pretrained(
            "./",
            checkpoint_file=model_path,
            archive_map={},
            data_name_or_path=dictionary_path,
            task="multilingual_translation"
        )

        sp_models = {
            lang: SentencePieceProcessor(
                model_file=f"{sentencepiece_prefix}.{lang}.model"
            ) for lang in x["task"].langs
        }

        return cls(
            models=x["models"],
            task=x["task"],
            cfg=x["args"],
            sp_models=sp_models,
        )

    @property
    def device(self):
        return self._float_tensor.device

    def binarize(self, sentence: str, language: str) -> LongTensor:
        return self.dicts[language].encode_line(sentence, add_if_not_exist=False).long()

    def apply_bpe(self, sentence: str, language: str) -> str:
        return " ".join(self.sp_models[language].encode(sentence, out_type=str))

    def string(self, tokens: Tensor, language: str) -> str:
        return self.dicts[language].string(tokens)

    @staticmethod
    def remove_bpe(sentence: str) -> str:
        return sentence.replace(" ", "").replace("\u2581", " ").strip()

    def encode(self, sentence: str, language: str) -> LongTensor:
        return self.binarize(self.apply_bpe(sentence, language), language)

    def decode(self, tokens: Tensor, language: str) -> str:
        return self.remove_bpe(self.string(tokens, language))

    def translate(
            self,
            sentences: List[str],
            src_language: str,
            tgt_language: str,
            beam: int = 5,
            max_sentences: Optional[int] = 10,
            max_tokens: Optional[int] = 1000,
    ) -> List[str]:
        """
        :param sentences: list of sentences to be translated
        :param src_language: source language
        :param tgt_language: target language
        :param beam: beam size for the beam search algorithm (decoding)
        :param max_sentences: max number of sentences in each batch
        :param max_tokens: max number of tokens in each batch, all sentences must be shorter than max_tokens.
        :return: list of translations corresponding to the input sentences
        """
        tokenized_sentences = [self.encode(sentence, src_language) for sentence in sentences]
        batched_hypos = self._generate(
            tokenized_sentences,
            src_language,
            tgt_language,
            beam=beam,
            max_sentences=max_sentences,
            max_tokens=max_tokens
        )
        return [self.decode(hypos[0]["tokens"], tgt_language) for hypos in batched_hypos]

    def _generate(
            self,
            tokenized_sentences: List[LongTensor],
            src_lang: str,
            tgt_lang: str,
            beam: int = 5,
            max_sentences: Optional[int] = 10,
            max_tokens: Optional[int] = None,
            skip_invalid_size_inputs=False,
    ) -> List[List[Dict[str, Tensor]]]:
        gen_args = copy.deepcopy(self.cfg.generation)
        with open_dict(gen_args):
            gen_args.beam = beam
        generator = self._build_generator(src_lang, tgt_lang, gen_args)

        results = []
        for batch in self._build_batches(
                tokenized_sentences,
                src_lang,
                tgt_lang,
                skip_invalid_size_inputs=skip_invalid_size_inputs,
                max_sentences=max_sentences,
                max_tokens=max_tokens
        ):
            batch = utils.apply_to_sample(lambda t: t.to(self.device), batch)
            translations = self.task.inference_step(
                generator, self.models, batch
            )
            for id, hypos in zip(batch["id"].tolist(), translations):
                results.append((id, hypos))

        # sort output to match input order
        outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]

        return outputs

    def _build_dataset_for_inference(
            self, src_tokens: List[LongTensor],
            src_lengths: LongTensor,
            src_lang: str,
            tgt_lang: str,
    ) -> FairseqDataset:
        return self.task.alter_dataset_langtok(
            LanguagePairDataset(
                src_tokens, src_lengths, self.dicts[src_lang]
            ),
            src_eos=self.dicts[src_lang].eos(),
            src_lang=src_lang,
            tgt_eos=self.dicts[tgt_lang].eos(),
            tgt_lang=tgt_lang,
        )

    def _build_batches(
            self,
            tokens: List[LongTensor],
            src_lang: str,
            tgt_lang: str,
            skip_invalid_size_inputs: bool,
            max_sentences: Optional[int] = 10,
            max_tokens: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        lengths = LongTensor([t.numel() for t in tokens])
        batch_iterator = self.task.get_batch_iterator(
            dataset=self._build_dataset_for_inference(tokens, lengths, src_lang, tgt_lang),
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            max_positions=self.max_positions[f"{src_lang}-{tgt_lang}"],
            ignore_invalid_inputs=skip_invalid_size_inputs,
            disable_iterator_cache=True,
        ).next_epoch_itr(shuffle=False)
        return batch_iterator

    def _build_generator(self, src_lang, tgt_lang, args):
        return SequenceGenerator(
            ModuleList([model.models[f"{src_lang}-{tgt_lang}"] for model in self.models]),
            self.dicts[tgt_lang],
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search.BeamSearch(self.dicts[tgt_lang]),
        )
