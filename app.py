import logging
from flask import request
from flask_cors import CORS

from nauron import Nauron

import settings
from nmt_worker import TranslationWorker

logger = logging.getLogger("gunicorn.error")

# Define application
app = Nauron(__name__)
CORS(app)

nmt = app.add_service(name=settings.SERVICE_NAME)

nmt.add_worker(TranslationWorker(nmt_model=settings.NMT_MODEL,
                                 spm_prefix=settings.SPM_MODEL_PREFIX,
                                 dict_path=settings.DICTIONARY_PATH,
                                 cpu=settings.CPU,
                                 factors=settings.FACTORS,
                                 max_sentences=settings.MAX_SENTS,
                                 max_tokens=settings.MAX_TOKENS,
                                 beam_size=settings.BEAM,
                                 lid_model=settings.LID_MODEL))

@app.post('/translation')
def translate():
    response = nmt.process_request(content=request.json)
    return response


if __name__ == '__main__':
    app.run(threaded=False)