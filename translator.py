"""Sockeye model loading and inference"""
import json
import logging

from typing import List

import mxnet as mx
import sentencepiece as spm
from sockeye.translate import inference

from truecaser import applytc


logger = logging.getLogger("nmt_service")


class Translator:
    models = None
    spm_model = spm.SentencePieceProcessor()
    tc_model = None

    def __init__(self, sockeye_model_folder: str, spm_model_path: str, tc_model_path: str, use_cpu: bool,
                 factors: dict):
        self.mx_ctx = mx.cpu() if use_cpu else mx.gpu()
        self.use_truecaser = True if tc_model_path is not None else False
        self.constrains = None  # Legacy thing
        self.factors = factors
        self.load_models(sockeye_model_folder, spm_model_path, tc_model_path)


    def _preprocess(self, sent: str, sent_factors: dict):
        sentence = applytc.processLine(self.tc_model, sent) if self.tc_model else sent

        reformat_sentence = " ".join(self.spm_model.encode(sentence, out_type=str))

        n_words = len(reformat_sentence.split())

        # Prepare factors
        factors = []
        if len(self.factors["sequence"]) > 0:
            for fs in self.factors["sequence"]:
                if fs in sent_factors:
                    factors.append(" ".join([sent_factors[fs]] * n_words))
                else:
                    factors.append(" ".join([f"{fs}0"] * n_words))

        preproc_json = {"text": reformat_sentence, "factors": factors}
        logger.info(f"Preprocessed: {sent} into {reformat_sentence}.")
        return json.dumps(preproc_json)

    def _postprocess(self, sent: str):
        postproc_sent = self.spm_model.DecodePieces(sent.split())
        postproc_sent = postproc_sent.replace("▁", " ").split()

        # Do truecasing
        postproc_sent[0] = postproc_sent[0].capitalize()

        res = " ".join(postproc_sent)

        logger.info(f"Postprocessed: {sent} into: {res}.")
        return res

    def _forward(self, sents: List[str]):
        translation_inputs = [
            inference.make_input_from_json_string(sentence_id=i, json_string=sent, translator=self.models)
            for (i, sent) in enumerate(sents)]
        outputs = self.models.translate(translation_inputs)
        return [(output.translation, output.score) for output in outputs]

    def translate(self, sents: List[str], sent_factors: dict):
        inputs = [self._preprocess(sent, sent_factors) for sent in sents]

        scored_translations = self._forward(inputs)

        translations, scores = zip(*scored_translations)

        postproc_translations = [self._postprocess(sent) for sent in translations]
        return postproc_translations, scores, inputs, translations

    def load_models(self, sockeye_model_folder: str, spm_model_path: str, tc_model_path: str):
        """Load translation and segmentation models."""

        self.models = load_sockeye_v1_translator_models([sockeye_model_folder, ], self.mx_ctx)
        self.spm_model.Load(spm_model_path)
        if tc_model_path:
            self.tc_model = applytc.loadModel(tc_model_path)


def load_sockeye_v1_translator_models(model_folders, ctx=mx.gpu()):
    logger.info(f"Loading Sockeye models. MXNET context: {ctx}")
    # TODO move beam size, batch size, etc to config file
    models, source_vocabs, target_vocab = inference.load_models(
        context=ctx,
        max_input_len=None,
        beam_size=3,
        batch_size=16,
        model_folders=model_folders,
        checkpoints=None,
        softmax_temperature=None,
        max_output_length_num_stds=2,
        decoder_return_logit_inputs=False,
        cache_output_layer_w_b=False)

    return inference.Translator(context=ctx,
                                ensemble_mode="linear",
                                bucket_source_width=10,
                                length_penalty=inference.LengthPenalty(1.0, 0.0),
                                beam_prune=0,
                                beam_search_stop='all',
                                models=models,
                                source_vocabs=source_vocabs,
                                target_vocab=target_vocab,
                                restrict_lexicon=None,
                                store_beam=False,
                                strip_unknown_words=False)






