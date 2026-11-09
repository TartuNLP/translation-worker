FROM python:3.10

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        gcc \
        g++ \
        libffi-dev \
        musl-dev \
        git

ENV PYTHONIOENCODING=utf-8
ENV MKL_NUM_THREADS=16

WORKDIR /app

RUN adduser --uid 1000 --disabled-password --gecos "app" app && \
    chown -R app:app /app
# Numeric so that Kubernetes runAsNonRoot can verify it
USER 1000

ENV PATH="/home/app/.local/bin:${PATH}"

COPY --chown=app:app requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt && \
    rm requirements.txt && \
    python -c "import nltk; nltk.download(\"punkt_tab\")"

COPY --chown=app:app . .

ENTRYPOINT ["python", "main.py"]
