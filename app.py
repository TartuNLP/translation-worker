import logging
from flask import request
from flask_cors import CORS

from nauron import Nauron

import settings

logger = logging.getLogger("gunicorn.error")

# Define application
app = Nauron(__name__)
CORS(app)

nmt = app.add_service(name=settings.SERVICE_NAME)

from nmt_worker import TranslationWorker

nmt.add_worker(TranslationWorker(nmt_model=settings.NMT_MODEL,
                                  spm_model=settings.SPM_MODEL,
                                  tc_model=settings.TC_MODEL,
                                  cpu=settings.CPU,
                                  factors=settings.FACTORS,
                                  char_limit=settings.CHAR_LIMIT))

@app.post('/translation')
def translate():
    response = nmt.process_request(content=request.json)
    return response


if __name__ == '__main__':
    app.run(threaded=False)