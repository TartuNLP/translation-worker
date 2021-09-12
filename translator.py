"""Sockeye model loading and inference"""
import logging
from typing import List
import torch

from interface import ModularHubInterface

logger = logging.getLogger("nmt")


class Translator:
    def __init__(
            self,
            fairseq_model_path: str,
            spm_prefix: str,
            dict_dir_path: str,
            languages: List[str],
            beam_size: int = 5,
            max_sentences: int = 10,
            max_tokens: int = 3000
    ):
        self.languages = languages

        self.beam_size = beam_size
        self.max_sentences = max_sentences
        self.max_tokens = max_tokens

        self.model = ModularHubInterface.from_pretrained(fairseq_model_path, spm_prefix, dict_dir_path)
        if torch.cuda.is_available():
            self.model.cuda()

    def translate(self, sents: List[str], src_lang: str, tgt_lang: str):
        return self.model.translate(sents, src_language=src_lang, tgt_language=tgt_lang,
                                    max_sentences=self.max_sentences, max_tokens=self.max_tokens, beam=self.beam_size)
