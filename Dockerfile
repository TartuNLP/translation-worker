FROM continuumio/miniconda3
RUN apt-get update && apt-get install -y build-essential

WORKDIR /app
COPY environment.yml .

# Create conda environment and configure the shell to use it
# Info: https://pythonspeed.com/articles/activate-conda-dockerfile/
RUN conda env create -f environment.yml -n nmt && rm environment.yml
SHELL ["conda", "run", "-n", "nmt", "/bin/bash", "-c"]
RUN python -c "import nltk; nltk.download(\"punkt\"); nltk.download(\"cmudict\")"
RUN pip install sockeye==1.18.106 --no-deps
SHELL ["/bin/bash", "-c"]

COPY . .
VOLUME /app/models

ENTRYPOINT conda run --no-capture-output -n nmt \
python nmt_worker.py