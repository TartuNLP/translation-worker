import json
import logging
from typing import List

import mxnet as mx
import mxnet.runtime
import sentencepiece as spm
from sockeye.translate import inference

from truecaser import applytc

logger = logging.getLogger("nmt")


class Translator:
    nmt_model = None
    spm_model = spm.SentencePieceProcessor()
    tc_model = None
    factor_sequence = None
    factors = None

    def __init__(self, nmt_model: str, spm_model: str, tc_model: str, factor_sequence: list, factors: dict):
        self.factor_sequence = factor_sequence
        self.factors = {}
        for factor, mapping in factors.items():
            self.factors[factor] = {}
            for k, v in mapping.items():
                self.factors[factor][k] = v
                self.factors[factor][v] = v
        self._load_models(nmt_model, spm_model, tc_model)

    def _load_models(self, nmt_model: str, spm_model: str, tc_model: str):
        """Load translation and segmentation models."""

        features = mxnet.runtime.Features()
        ctx = mx.gpu() if features.is_enabled("CUDA") and mx.context.num_gpus() else mx.cpu()
        logger.info(f"Loading Sockeye models. MXNET context: {ctx}")

        # TODO move beam size, batch size, etc to config file
        models, source_vocabs, target_vocab = inference.load_models(
            context=ctx,
            max_input_len=None,
            beam_size=3,
            batch_size=16,
            model_folders=[nmt_model],
            checkpoints=None,
            softmax_temperature=None,
            max_output_length_num_stds=2,
            decoder_return_logit_inputs=False,
            cache_output_layer_w_b=False)

        self.nmt_model = inference.Translator(
            context=ctx,
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
            strip_unknown_words=False
        )
        self.spm_model.Load(spm_model)
        if tc_model:
            self.tc_model = applytc.loadModel(tc_model)

    def _preprocess(self, sentence: str, sentence_factors: dict):
        tc_sentence = applytc.processLine(self.tc_model, sentence) if self.tc_model else sentence

        reformat_sentence = " ".join(self.spm_model.encode(tc_sentence, out_type=str))
        n_words = len(reformat_sentence.split())

        # Prepare factors
        factors = []
        for factor in self.factor_sequence:
            if factor in sentence_factors:
                factors.append(" ".join([sentence_factors[factor]] * n_words))
            else:
                factors.append(" ".join([f"{factor}0"] * n_words))

        preproc_json = {"text": reformat_sentence, "factors": factors}
        logger.info(f"Preprocessed: {sentence} into {reformat_sentence}.")
        return json.dumps(preproc_json)

    def _postprocess(self, sent: str):
        postproc_sent = self.spm_model.DecodePieces(sent.split())
        postproc_sent = postproc_sent.replace("▁", " ").split()

        # Truecase
        if postproc_sent:
            postproc_sent[0] = postproc_sent[0].capitalize()

        res = " ".join(postproc_sent)

        logger.info(f"Postprocessed: {sent} into: {res}.")
        return res

    def _forward(self, sentences: List[str]):
        translation_inputs = [
            inference.make_input_from_json_string(sentence_id=i, json_string=sentence, translator=self.nmt_model)
            for (i, sentence) in enumerate(sentences)]
        outputs = self.nmt_model.translate(translation_inputs)
        translations = [output.translation for output in outputs]
        scores = [output.score for output in outputs]

        return translations, scores

    def translate(self, sentences: List[str], sentence_factors: dict):
        inputs = [self._preprocess(sentence, sentence_factors) for sentence in sentences]

        translations, scores = self._forward(inputs)

        postproc_translations = [self._postprocess(translation) for translation in translations]
        return postproc_translations, scores, inputs, translations
