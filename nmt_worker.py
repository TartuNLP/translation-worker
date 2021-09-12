import itertools
import logging
from typing import Dict, Any, Optional, List, Union

from nltk import sent_tokenize
from marshmallow import Schema, fields, validate, ValidationError
from nauron import Response, Worker

import settings
from translator import Translator

logger = logging.getLogger("nmt")


class TranslationWorker(Worker):
    engine: Translator = None

    def __init__(self, nmt_model: str, spm_model: str, tc_model: str, factor_sequence: list, factors: dict,
                 defaults: dict, char_limit: int = 10000):

        self.engine = Translator(nmt_model, spm_model, tc_model, factor_sequence, factors)
        logger.info("All models loaded")

        class NMTSchema(Schema):
            text = fields.Raw(required=True, validate=(lambda obj: type(obj) in [str, list]))
            src = fields.Str(missing=None)
            tgt = fields.Str(required=True, validate=validate.OneOf(self.engine.factors['lang'].keys()))
            domain = fields.Str(missing=None)
            application = fields.Str(missing=None)

        self.schema = NMTSchema
        self.char_limit = char_limit
        self.defaults = defaults

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
                        f"tgt: {body['tgt']}, "
                        f"domain: {body['domain']}}}")
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
            sent_factors = {'lang': self.engine.factors['lang'][body['tgt']]}
            if 'domain' in self.engine.factor_sequence:
                if body['domain'] is not None and body['domain'] in self.engine.factors['domain']:
                    sent_factors['domain'] = self.engine.factors['domain'][body['domain']]
                else:
                    sent_factors['domain'] = self.defaults['domain']

            translations, _, _, _ = self.engine.translate(sentences, sent_factors)
            if delimiters:
                translations = ''.join(itertools.chain.from_iterable(zip(delimiters, translations))) + delimiters[-1]

            return Response({'result': translations}, mimetype="application/json")


if __name__ == "__main__":
    worker = TranslationWorker(**settings.WORKER_PARAMETERS)
    worker.start(connection_parameters=settings.MQ_PARAMETERS,
                 service_name=settings.SERVICE_NAME,
                 routing_key=settings.ROUTES[0],
                 alt_routes=settings.ROUTES[1:])
