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
    def __init__(self, nmt_model, spm_prefix, dict_path, cpu, factors, max_sentences, max_tokens, beam_size, char_limit: int = 10000):
        self.engine = Translator(fairseq_model_path=nmt_model,
                                 spm_prefix=spm_prefix,
                                 dict_dir_path=dict_path,
                                 use_cpu=cpu,
                                 factors=factors,
                                 max_sentences=max_sentences,
                                 max_tokens=max_tokens,
                                 beam_size=beam_size
                                 )
        logger.info("All models loaded")

        class NMTSchema(Schema):
            text = fields.Raw(validate=(lambda obj: type(obj) in [str, list]))
            src = fields.Str(missing=factors['lang']['factors'][0],
                             validate=validate.OneOf(factors['lang']['mapping'].keys()))
            tgt = fields.Str(missing=factors['lang']['factors'][0],
                             validate=validate.OneOf(factors['lang']['mapping'].keys()))
            domain = fields.Str(missing="")
            application = fields.Str(missing=None)

        self.schema = NMTSchema
        self.char_limit = char_limit

    def process_request(self, body: Dict[str, Any], _: Optional[str] = None) -> Response:
        try:
            body = self.schema().load(body)
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
            sent_factors = {'src_lang': self.engine.factors['lang']['mapping'][body['src']],
                            'tgt_lang': self.engine.factors['lang']['mapping'][body['tgt']]}
            if 'domain' in self.engine.factors:
                if body['domain'] and body['domain'] in self.engine.factors['domain']['mapping']:
                    sent_factors['domain'] = self.engine.factors['domain']['mapping'][body['domain']]
                else:
                    sent_factors['domain'] = self.engine.factors['domain']['factors'][0]
            translations = self.engine.translate(sentences, sent_factors)
            if delimiters:
                translations = ''.join(itertools.chain.from_iterable(zip(delimiters, translations))) + delimiters[-1]

            return Response({'result': translations}, mimetype="application/json")


if __name__ == "__main__":
    mq_parameters = settings.MQ_PARAMS
    worker = TranslationWorker(nmt_model=settings.NMT_MODEL,
                               spm_prefix=settings.SPM_MODEL_PREFIX,
                               dict_path=settings.DICTIONARY_PATH,
                               cpu=settings.CPU,
                               factors=settings.FACTORS,
                               max_sentences=settings.MAX_SENTS,
                               max_tokens=settings.MAX_TOKENS,
                               beam_size=settings.BEAM)
    worker.start(connection_parameters=mq_parameters,
                 service_name=settings.SERVICE_NAME,
                 routing_key=settings.ROUTING_KEY,
                 alt_routes=settings.MQ_ALT_ROUTES)
