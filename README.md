# TartuNLP translation worker

This repository contains TartuNLP's machine translation models and workers to run them and 
process requests from RabbitMQ. The current main model supports seven languages: Estonian, English, Russian, German, 
Finnish, Latvian and Lithuanian. Additionally, there is a Finno-Ugric model (supporting Estonian, Finnish, Võro, 
Northern Sami and Southern Sami).

The project is developed by the [NLP research group](https://tartunlp.ai) at the [University of Tartu](https://ut.ee).
Neural machine translation can also be tested in our [web demo](https://translate.ut.ee/).

### Configuration and model files

The models can be downloaded from the [releases](https://github.com/TartuNLP/translation-worker/releases) section of 
this repository. If a release does not specify the model information, the model from the previous release can be 
used. We advise always using the latest available version to ensure best model quality and code compatibility. The 
default files for running these models are provided in the `config/` directory.

### Docker setup

[Docker images](https://ghcr.io/tartunlp/translation-worker) for the workers are published alongside this repository. 
Each image version correlates to a specific release. The required model file(s) are excluded from the image to 
reduce the image size and should be downloaded from the releases section and their directory should be attached to the 
volume `/app/models`.

Logs are stored in `/app/logs/` and logging configuration is loaded from `/app/config/logging.ini`. Service
configuration from `/app/config/config.yaml` files.

The RabbitMQ connection parameters are set with environment variables, exchange and queue names are dependent on the
`service` and `route` values in `config.yaml`. The setup can be tested with the following sample
`docker-compose.yml` configuration where `WORKER_NAME` matches the name in your config file. One worker should
be added for each model.

```yaml
version: '3'
services:
  rabbitmq:
    image: 'rabbitmq:3.6-alpine'
    environment:
      - RABBITMQ_DEFAULT_USER=${RABBITMQ_USER}
      - RABBITMQ_DEFAULT_PASS=${RABBITMQ_PASS}
  nmt_api:
    image: ghcr.io/tartunlp/translation-api:latest
    environment:
      - MQ_HOST=rabbitmq
      - MQ_PORT=5672
      - MQ_USERNAME=${RABBITMQ_USER}
      - MQ_PASSWORD=${RABBITMQ_PASS}
    ports:
      - '80:8000'
    depends_on:
      - rabbitmq
  nmt_worker_smugri:
    image: ghcr.io/tartunlp/translation-worker:latest
    environment:
      - MQ_HOST=rabbitmq
      - MQ_PORT=5672
      - MQ_USERNAME=${RABBITMQ_USER}
      - MQ_PASSWORD=${RABBITMQ_PASS}
    volumes:
      - ./models:/app/models
    depends_on:
      - rabbitmq
    command: ["--model-name", "smugri"]
```

### Manual setup

The following steps have been tested on Ubuntu. The code is both CPU and GPU compatible (CUDA required), but the
`environment.gpu.yml` file should be used for a GPU installation.

- Make sure you have the following prerequisites installed:
    - Conda (see https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
    - GNU Compiler Collection (`sudo apt install build-essential`)

- Clone this repository with submodules
- Create and activate a Conda environment with all dependencies:

```commandline
conda env create -f environments/environment.yml -n nmt
conda activate nmt
pip install sockeye==1.18.106 --no-deps
python -c "import nltk; nltk.download(\"punkt\")"
```

- Download the models from the [releases section](https://github.com/TartuNLP/translation-worker/releases) and
  place inside the `models/` directory.
- Check the configuration files and change any defaults as needed. Make sure that the `checkpoint` parameters in
  `config/config.yaml` points to the model filse you just downloaded. By default, logs will be stored in the
  `logs/` directory which is specified in the `config/logging.ini` file.
- Specify RabbitMQ connection parameters with environment variables or in a `config/.env` file as illustrated in the
  `config/sample.env`.

Run the worker with the following command where `$MODEL_NAME` matches the worker name in your config file:
```commandline
python main.py --model-name $MODEL_NAME [--log-config config/logging.prod.ini --config config/config.yaml]
```

