""" Simple Flask demo app, to run everything locally. """
import logging

from flask import Flask
from flask_cors import CORS

from nauron import Endpoint, ServiceConf

import settings
from translator import Translator
from nmt_service import TranslationService

logger = logging.getLogger("nmt_service")

# Define Flask application
app = Flask(__name__)
CORS(app)

factors = settings.FACTORS

translation_engine = Translator(settings.NMT_MODEL, settings.SPM_MODEL_PREFIX, settings.DICTIONARY_PATH,
                                    settings.CPU, settings.FACTORS, max_sentences=settings.MAX_SENTS,
                                    max_tokens=settings.MAX_TOKENS, beam_size=settings.BEAM)
logger.info("All models loaded")

service_conf = ServiceConf(name='translation',
                           endpoint='/translation',
                           engines= {'public': TranslationService(translation_engine, settings.CHAR_LIMIT)})

# Define API endpoints
app.add_url_rule(service_conf.endpoint, view_func=Endpoint.as_view(service_conf.name, service_conf))


if __name__ == '__main__':
    app.run(threaded=False)