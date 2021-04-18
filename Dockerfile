FROM continuumio/miniconda3
RUN apt-get update && apt-get install -y build-essential

WORKDIR /var/log/nmt
WORKDIR /nmt
COPY . .

# Create conda environment and configure the shell to use it
# Info: https://pythonspeed.com/articles/activate-conda-dockerfile/
RUN conda env create -f environment.yml -n nmt && rm environment.yml
SHELL ["conda", "run", "-n", "nmt", "/bin/bash", "-c"]
# TODO remove
RUN pip install -e nauron/ && \
python -c "import nltk; nltk.download(\"punkt\"); nltk.download(\"cmudict\")"
RUN pip install sockeye==1.18.106 --no-deps
SHELL ["/bin/bash", "-c"]

VOLUME /nmt/models

ENTRYPOINT conda run --no-capture-output -n nmt \
python nmt_service.py