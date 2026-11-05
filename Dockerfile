FROM vllm/vllm-openai:v0.33.0

ENV PYTHONIOENCODING=utf-8
ENV NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt && \
    python3 -c "import nltk; nltk.download('punkt_tab', download_dir='$NLTK_DATA')"

RUN useradd --create-home app && \
    chown -R app:app /app
USER app

COPY --chown=app:app . .

ENTRYPOINT ["python3", "main.py"]
