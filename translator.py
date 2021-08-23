"""Sockeye model loading and inference"""
import logging

from typing import List

from interface import ModularHubInterface

logger = logging.getLogger("nmt_service")


class Translator:
    def __init__(
            self,
            fairseq_model_path: str,
            spm_prefix: str,
            dict_dir_path: str,
            use_cpu: bool,
            factors: dict,
            beam_size: int = 5,
            max_sentences: int = 10,
            max_tokens: int = 3000
    ):
        self.factors = factors

        self.beam_size = beam_size
        self.max_sentences = max_sentences
        self.max_tokens = max_tokens

        self.model = ModularHubInterface.from_pretrained(fairseq_model_path, spm_prefix, dict_dir_path)
        if not use_cpu:
            self.model.cuda()

    def translate(self, sents: List[str], sent_factors: dict):
        return self.model.translate(sents, src_language=sent_factors["src_lang"], tgt_language=sent_factors["tgt_lang"],
                                    max_sentences=self.max_sentences, max_tokens=self.max_tokens, beam=self.beam_size)
