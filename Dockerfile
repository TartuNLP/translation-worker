FROM continuumio/miniconda3
RUN apt-get update && apt-get install -y build-essential

ENV PYTHONIOENCODING=utf-8
VOLUME /app/models

COPY environments/environment.yml .
RUN conda env create -f environment.yml -n nmt && rm environment.yml
SHELL ["conda", "run", "-n", "nmt", "/bin/bash", "-c"]
RUN python -c "import nltk; nltk.download(\"punkt\"); nltk.download(\"cmudict\")"
RUN pip install sockeye==1.18.106 --no-deps
SHELL ["/bin/bash", "-c"]

WORKDIR /app
RUN adduser --disabled-password --gecos "app" app && \
    chown -R app:app /app && \
    chown -R app:app /opt/conda/envs/nmt

USER app

COPY --chown=app:app . .

ENV WORKER_NAME=""
RUN echo "python nmt_worker.py --worker \$WORKER_NAME" > entrypoint.sh

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "nmt", "bash", "entrypoint.sh"]
