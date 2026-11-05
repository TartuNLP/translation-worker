# TartuNLP translation worker

This repository contains code for running TartuNLP's LLM-based multilingual machine translation models. This branch
runs Apertus 1.5 based models with [vLLM](https://github.com/vllm-project/vllm) and requires an NVIDIA GPU. The models
can be run locally to translate plain text files or as worker that process requests from RabbitMQ. The workers are compatible with
our [translation API](https://github.com/TartuNLP/translation-api).

The project is developed by the [NLP research group](https://tartunlp.ai) at the [University of Tartu](https://ut.ee).
Neural machine translation can also be tested in our [web demo](https://translate.ut.ee/).

### Configuration and model files

The worker runs Apertus models in the Hugging Face format. Apertus 1.5 (`swiss-ai/Apertus-v1.5-8B` and models
fine-tuned from it) is a multimodal model that is not yet supported by released vLLM and transformers versions. Its
text-only part is however exactly an Apertus 1 model, so it is exported as a plain `ApertusForCausalLM` checkpoint
first (requires `torch` and `safetensors`):

```commandline
hf download swiss-ai/Apertus-v1.5-8B --local-dir apertus-1.5-src
python scripts/export_apertus15_text.py apertus-1.5-src models/apertus
```

The export drops the image and audio tokenizers, their embedding rows and their tokens from the tokenizer, and gives
identical results to the original model on text input.

The bf16 model needs ~16 GB of GPU memory for the weights. For smaller GPUs (e.g. 10-12 GB), the model can be
quantized to 4-bit weights with [AutoRound](https://github.com/intel/auto-round) (~6 GB, ~15-20 minutes on an
RTX 3080). The calibration file should contain a few hundred translation examples covering all translation
directions, ideally sampled from the model's training data (see the script for the format):

```commandline
pip install auto-round==0.16.1
python scripts/quantize_autoround.py models/apertus models/apertus-w4a16 --calibration calibration.jsonl
```

The default config for running the model is specified in the included `config/config.yaml`. The `prompt_template`
and the language names in `language_codes` must match the ones used in training. `model_path` points to the model
directory:

```
models/
└── apertus/
    ├── config.json
    ├── generation_config.json
    ├── model-0000X-of-0000Y.safetensors
    ├── model.safetensors.index.json
    ├── chat_template.jinja
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── tokenizer.json
```

### Docker setup

[Docker images](https://ghcr.io/tartunlp/translation-worker) for the workers are published alongside this repository.
Each image version correlates to a specific release. The required model file(s) are excluded from the image to reduce
the image size and should be attached at `/app/models`.

Logging configuration is loaded from `/app/config/logging.prod.ini` and service configuration from
the `/app/config/config.yaml` file. The included config is commented to illustrate how new model configurations could be
added.

The following environment variables should be configured when running the container:

- `MQ_USERNAME` - RabbitMQ username
- `MQ_PASSWORD` - RabbitMQ user password
- `MQ_HOST` - RabbitMQ host
- `MQ_PORT` (optional) - RabbitMQ port (`5672` by default)
- `MQ_EXCHANGE` (optional) - RabbitMQ exchange name (`translation` by default)
- `MQ_HEARTBEAT` (optional) - heartbeat interval (`60` seconds by default)
- `MQ_CONNECTION_NAME` (optional) - friendly connection name (`Translation worker` by default)

By default, the container entrypoint is `main.py` without additional arguments. The model configuration is loaded from
the config file specified by `--model-config`. The container needs access to a GPU (`docker run --gpus all`) and
enough shared memory for vLLM (`--ipc=host` or `--shm-size`). The full list of supported flags can be seen by running `python main.py -h`:

```commandline
usage: main.py [-h] [--model-config MODEL_CONFIG] [--log-config LOG_CONFIG] [--input-file INPUT_FILE] [--output-file OUTPUT_FILE] [--input-lang INPUT_LANG] [--output-lang OUTPUT_LANG]

A neural machine translation engine. This application supports two modes of operation: 
    a) a worker that processes incoming translation requests via RabbitMQ;
    b) translation of a text file into a new file;

optional arguments:
  -h, --help            show this help message and exit
  --model-config MODEL_CONFIG
                        The model config YAML file to load. (default: config/config.yaml)
  --log-config LOG_CONFIG
                        Path to log config file. (default: config/logging.prod.ini)

Local file translation arguments, if the following arguments exist, local file translation is started. Otherwise a RabbitMQ worker is started:
  --input-file INPUT_FILE
                        Path to the input text file. (default: None)
  --output-file OUTPUT_FILE
                        Path to the output text file. (default: None)
  --input-lang INPUT_LANG
                        Input language code (default: None)
  --output-lang OUTPUT_LANG
                        Output language code. (default: None)
```

The setup can be tested with the following sample `docker-compose.yml` configuration:

``` yaml
version: '3'
services:
  rabbitmq:
    image: 'rabbitmq'
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
  nmt_worker:
    image: ghcr.io/tartunlp/translation-worker:latest
    environment:
      - MQ_HOST=rabbitmq
      - MQ_PORT=5672
      - MQ_USERNAME=${RABBITMQ_USER}
      - MQ_PASSWORD=${RABBITMQ_PASS}
    volumes:
      - ./models:/app/models
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [ gpu ]
    depends_on:
      - rabbitmq
```

### Manual setup

The following steps have been tested on Linux with Python 3.12 and an NVIDIA GPU.

- Clone this repository
- Install the dependencies in a clean environment:
  ```commandline
  pip install -r requirements.txt
  python -c "import nltk; nltk.download('punkt_tab')"
  ```
- Prepare the model as described in [configuration and model files](#configuration-and-model-files).
- Check the configuration files and change any defaults as needed. Make sure that the paths in
  `config/config.yaml` points to the model files you just prepared.
- Specify RabbitMQ connection parameters with environment variables or in a `config/.env` file as illustrated in the
  `config/sample.env`.

Run the worker with the following command:

```commandline
python main.py [--log-config config/logging.ini --model-config config/config.yaml]
```

Or run local file translation with the following command:

```commandline
python main.py --input-file input.txt --output-file output.txt --input-lang est --output-lang eng [--log-config config/logging.ini --model-config config/config.yaml]
```

### Performance and Hardware Requirements

The worker requires an NVIDIA GPU. vLLM allocates `gpu_memory_utilization` (in `config/config.yaml`) of the GPU memory
at startup, for the model weights and the KV cache. The bf16 8B model needs a GPU with at least 24 GB of memory, the
4-bit quantized model runs on 10 GB GPUs (tested on an RTX 3080 with `max_model_len: 4096`).

All sentences of a request are translated in one batch. `max_num_seqs` limits the number of sentences that are
processed in parallel, and `max_model_len` the length of a prompt and its translation.
