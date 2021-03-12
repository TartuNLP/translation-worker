import itertools
import logging

from typing import Dict, Any, Optional

from nltk import sent_tokenize
from marshmallow import Schema, fields, validate, ValidationError

from nauron import Response, Service, MQConsumer

import settings
from translator import Translator

logger = logging.getLogger("nmt_service")

class TranslationService(Service):
    def __init__(self, engine: Translator, char_limit: int = 10000):
        class NMTSchema(Schema):
            text = fields.Raw(validate=(lambda obj: type(obj) in [str, list]))
            src = fields.Str()
            tgt = fields.Str(missing=engine.factors['lang']['factors'][0],
                             validate=validate.OneOf(engine.factors['lang']['mapping'].keys()))
            domain = fields.Str(missing="")

        self.schema = NMTSchema
        self.char_limit = char_limit
        # Load model
        self.engine = engine

    def process_request(self, body: Dict[str, Any], _: Optional[str] = None) -> Response:
        try:
            body = self.schema().load(body)
        except ValidationError as error:
            return Response(content=error.messages, http_status_code=400)

        delimiters = None
        if type(body['text']) == str:
            text = body['text']
            sentences = sent_tokenize(body['text'])
            try:
                delimiters = []
                for sent in sentences:
                    idx = text.index(sent)
                    delimiters.append(text[:idx])
                    text = text[idx + len(sent):]
                delimiters.append(text)
            except ValueError:
                delimiters = ['', *[' ' for _ in range(len(sentences)-1)], '']
        else:
            sentences = body['text']

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
                translations = ''.join(itertools.chain.from_iterable(zip(delimiters, translations)))+delimiters[-1]

            return Response({'result': translations}, mimetype="application/json")


if __name__ == "__main__":
    factors = settings.FACTORS
    mq_parameters = settings.MQ_PARAMS

    translation_engine = Translator(settings.NMT_MODEL, settings.SPM_MODEL, settings.TC_MODEL, settings.CPU, factors)
    logger.info("All models loaded")

    worker = MQConsumer(service=TranslationService(translation_engine, settings.CHAR_LIMIT),
                        connection_parameters=mq_parameters,
                        exchange_name=settings.MQ_EXCHANGE,
                        queue_name=settings.MQ_QUEUE_NAME,
                        alt_routes=settings.MQ_ALT_ROUTES)
    worker.start()


