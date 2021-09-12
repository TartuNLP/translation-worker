import itertools
import logging
from typing import Dict, Any, Optional, List, Union

from nltk import sent_tokenize
from marshmallow import Schema, fields, validate, ValidationError
from nauron import Response, Worker
import fasttext

import settings
from translator import Translator

logger = logging.getLogger("nmt")


class TranslationWorker(Worker):
    def __init__(self, nmt_model, spm_prefix, dict_path, lid_model, languages, max_sentences, max_tokens, beam_size,
                 char_limit: int = 10000):
        self.translator = Translator(fairseq_model_path=nmt_model,
                                     spm_prefix=spm_prefix,
                                     dict_dir_path=dict_path,
                                     languages=languages.values(),
                                     max_sentences=max_sentences,
                                     max_tokens=max_tokens,
                                     beam_size=beam_size)
        self.lid_model = fasttext.load_model(lid_model)
        logger.info("All models loaded")

        self.languages = languages.copy()
        for language in languages.values():
            self.languages[language] = language
        self.default_language = list(self.languages.values())[0]

        class NMTSchema(Schema):
            text = fields.Raw(required=True, validate=(lambda obj: type(obj) in [str, list]))
            src = fields.Str(missing=None)
            tgt = fields.Str(required=True, validate=validate.OneOf(self.languages.keys()))
            domain = fields.Str(missing=None)
            application = fields.Str(missing=None)

        self.schema = NMTSchema
        self.char_limit = char_limit

    def _detect_language(self, sentences: List[str]) -> str:
        if self.lid_model is None:
            return self.default_language
        pred_lang = self.lid_model.predict(" ".join(sentences))[0][0][9:]
        if pred_lang in self.translator.languages:
            logger.info(f"Detected src={pred_lang}")
            return pred_lang
        else:
            logger.info(
                f"Detected src={pred_lang} not in supported languages. Defaulting to src={self.default_language}")
            return self.default_language

    @staticmethod
    def _sentence_tokenize(text: Union[str, List]) -> (List, Optional[List]):
        """
        Split text into sentences and save info about delimiters between them to restore linebreaks,
        whitespaces, etc.
        """
        delimiters = None
        if type(text) == str:
            sentences = [sent.strip() for sent in sent_tokenize(text)]
            try:
                delimiters = []
                for sentence in sentences:
                    idx = text.index(sentence)
                    delimiters.append(text[:idx])
                    text = text[idx + len(sentence):]
                delimiters.append(text)
            except ValueError:
                delimiters = ['', *[' ' for _ in range(len(sentences) - 1)], '']
        else:
            sentences = [sent.strip() for sent in text]

        return sentences, delimiters

    def process_request(self, body: Dict[str, Any], _: Optional[str] = None) -> Response:
        try:
            body = self.schema().load(body)
            logger.info(f"Request received: {{"
                        f"application: {body['application']}, "
                        f"src: {body['src']}, "
                        f"tgt: {body['tgt']}}}")
        except ValidationError as error:
            return Response(content=error.messages, http_status_code=400)

        sentences, delimiters = self._sentence_tokenize(body['text'])
        length = sum([len(sent) for sent in sentences])

        if length == 0:
            if type(body['text']) == str:
                return Response({'result': ""}, mimetype="application/json")
            else:
                return Response({'result': []}, mimetype="application/json")
        elif length > self.char_limit:
            return Response(content=f"Request is too large ({length} characters). "
                                    f"Maximum request size is {self.char_limit} characters.",
                            http_status_code=413)
        else:
            src_lang = self.languages[body['src']] if body['src'] is not None else self._detect_language(sentences)
            tgt_lang = self.languages[body['tgt']] if body['tgt'] is not None else self.default_language

            translations = self.translator.translate(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
            if delimiters:
                translations = ''.join(itertools.chain.from_iterable(zip(delimiters, translations))) + delimiters[-1]

            return Response({'result': translations}, mimetype="application/json")


if __name__ == "__main__":
    worker = TranslationWorker(**settings.WORKER_PARAMETERS)
    worker.start(connection_parameters=settings.MQ_PARAMETERS,
                 service_name=settings.SERVICE_NAME,
                 routing_key=settings.ROUTES[0],
                 alt_routes=settings.ROUTES[1:])
