import itertools
import logging

from typing import Dict, Any, Optional

from nltk import sent_tokenize
from marshmallow import Schema, fields, validate, ValidationError

from nauron import Response, Worker

import settings
from translator import Translator

logger = logging.getLogger("nmt_service")


class TranslationWorker(Worker):
    engine: Translator = None
    def __init__(self, nmt_model: str, spm_model: str, tc_model, cpu: bool, factors: dict, char_limit: int = 10000):
        self._init_translator(nmt_model, spm_model, tc_model, cpu, factors)
        logger.info("All models loaded")

        class NMTSchema(Schema):
            text = fields.Raw(validate=(lambda obj: type(obj) in [str, list]))
            src = fields.Str()
            tgt = fields.Str(missing=self.engine.factors['lang']['factors'][0],
                             validate=validate.OneOf(self.engine.factors['lang']['mapping'].keys()))
            domain = fields.Str(missing="")
            application = fields.Str(allow_none=True)

        self.schema = NMTSchema
        self.char_limit = char_limit

    def _init_translator(self, nmt_model, spm_model, tc_model, cpu, factors):
        self.engine = Translator(nmt_model, spm_model, tc_model, cpu, factors)

    def process_request(self, body: Dict[str, Any], _: Optional[str] = None) -> Response:
        try:
            body = self.schema().load(body)
            logger.info(f"Request source: {body['application']}")
        except ValidationError as error:
            return Response(content=error.messages, http_status_code=400)

        delimiters = None
        if type(body['text']) == str:
            text = body['text']
            sentences = [sent.strip() for sent in sent_tokenize(body['text'])]
            try:
                delimiters = []
                for sent in sentences:
                    idx = text.index(sent)
                    delimiters.append(text[:idx])
                    text = text[idx + len(sent):]
                delimiters.append(text)
            except ValueError:
                delimiters = ['', *[' ' for _ in range(len(sentences) - 1)], '']
        else:
            sentences = [sent.strip() for sent in body['text']]

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
            sent_factors = {'lang': self.engine.factors['lang']['mapping'][body['tgt']]}
            if 'domain' in self.engine.factors:
                if body['domain'] and body['domain'] in self.engine.factors['domain']['mapping']:
                    sent_factors['domain'] = self.engine.factors['domain']['mapping'][body['domain']]
                else:
                    sent_factors['domain'] = self.engine.factors['domain']['factors'][0]
            translations, _, _, _ = self.engine.translate(sentences, sent_factors)
            if delimiters:
                translations = ''.join(itertools.chain.from_iterable(zip(delimiters, translations))) + delimiters[-1]

            return Response({'result': translations}, mimetype="application/json")


if __name__ == "__main__":
    mq_parameters = settings.MQ_PARAMS
    worker = TranslationWorker(nmt_model=settings.NMT_MODEL,
                               spm_model=settings.SPM_MODEL,
                               tc_model=settings.TC_MODEL,
                               cpu=settings.CPU,
                               factors=settings.FACTORS,
                               char_limit=settings.CHAR_LIMIT)
    worker.start(connection_parameters=mq_parameters,
                 service_name=settings.SERVICE_NAME,
                 routing_key=settings.ROUTING_KEY,
                 alt_routes=settings.MQ_ALT_ROUTES)
