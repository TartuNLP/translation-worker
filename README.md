# TartuNLP translation worker

This repository contains TartuNLP's modular multilingual machine translation models. The models can be run locally to
translate plain text files or as worker that process requests from RabbitMQ. This application is based on
a [custom version of FairSeq](https://github.com/TartuNLP/fairseq). The workers are compatible with
our [translation API](https://github.com/TartuNLP/translation-api).

The project is developed by the [NLP research group](https://tartunlp.ai) at the [University of Tartu](https://ut.ee).
Neural machine translation can also be tested in our [web demo](https://translate.ut.ee/).

### Configuration and model files

The models can be downloaded from our [HuggingFace](https://huggingface.co/tartuNLP), compatible models are marked with
the [modularNMT](https://huggingface.co/models?other=modularNMT&pipeline_tag=translation&sort=downloads) tag. Older
models and compatible code versions can be found from the
[releases](https://github.com/TartuNLP/translation-worker/releases) section.

The default config for running these models is specified in the included `config/config.yaml`. For example, the first
model configuration corresponds to the following `models/` folder structure:

```
models/
└── septilang
    ├── modular_model.pt
    ├── dict.de.txt
    ├── dict.en.txt
    ├── dict.et.txt
    ├── dict.fi.txt
    ├── dict.lt.txt
    ├── dict.lv.txt
    ├── dict.ru.txt
    ├── sp-model.de.model
    ├── sp-model.en.model
    ├── sp-model.et.model
    ├── sp-model.fi.model
    ├── sp-model.lt.model
    ├── sp-model.lv.model
    └── sp-model.ru.model
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
- `MKL_NUM_THREADS` (optional) - number of threads used for intra-op parallelism by PyTorch. `16` by default. If set to
  a blank value, it defaults to the number of CPU cores which may cause computational overhead when deployed on larger
  nodes. Alternatively, the `docker run` flag `--cpuset-cpus` can be used to control this. For more details, refer to
  the [performance and hardware requirements](#performance-and-hardware-requirements) section below.

By default, the container entrypoint is `main.py` without additional arguments, but arguments should be defined with the
`COMMAND` option. The only required flag is `--model-name` to select which model is loaded by the worker. The full list
of supported flags can be seen by running `python main.py -h`:

```commandline
usage: main.py [-h] --model-name MODEL_NAME [--model-config MODEL_CONFIG] [--log-config LOG_CONFIG] [--input-file INPUT_FILE] [--output-file OUTPUT_FILE] [--input-lang INPUT_LANG] [--output-lang OUTPUT_LANG]

A neural machine translation engine. This application supports two modes of operation: 
    a) a worker that processes incoming translation requests via RabbitMQ;
    b) translation of a text file into a new file;

optional arguments:
  -h, --help            show this help message and exit
  --model-name MODEL_NAME
                        The model to load. Refers to the model name in the config file. (default: None)
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
      - MKL_NUM_THREADS=8
    volumes:
      - ./models:/app/models
    command: ["--model-name", "septilang"]
    depends_on:
      - rabbitmq
```

### Manual setup

The following steps have been tested on Ubuntu. The code is both CPU and GPU compatible (CUDA is required).

- Clone this repository
- Install prerequisites:
    - GNU Compiler Collection (`sudo apt install build-essential`)
    - For a **CPU** installation we recommend using the included `requirements.txt` file in a clean environment (tested
      with Python 3.10)
      ```commandline
      pip install -r requirements.txt
      ```

    - For a **GPU** installation, use the `environment.yml` file instead.
        - Make sure you have the following prerequisites installed:
            - CUDA (see https://developer.nvidia.com/cuda-downloads)
            - Conda (see https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

        - Then create and activate a Conda environment with all dependencies:
          ```commandline
          conda env create -f environment.yml -n nmt
          conda activate nmt
          python -c "import nltk; nltk.download(\"punkt\")"
          ```

- Download the models from
  the [HuggingFace](https://huggingface.co/models?other=modularNMT&pipeline_tag=translation&sort=downloads) and place
  inside the `models/` directory
- Check the configuration files and change any defaults as needed. Make sure that the paths in
  `config/config.yaml` points to the model files you just downloaded.
- Specify RabbitMQ connection parameters with environment variables or in a `config/.env` file as illustrated in the
  `config/sample.env`.

Run the worker with the following command where `$MODEL_NAME` matches the worker name in your config file:

```commandline
python main.py --model-name $MODEL_NAME [--log-config config/logging.ini --model-config config/config.yaml]
```

Or run local file translation with the following command:

```commandline
python main.py --model-name $MODEL_NAME --input-file input.txt --output-file output.txt --input-lang est --output-lang eng [--log-config config/logging.ini --model-config config/config.yaml]
```

### Performance and Hardware Requirements

When running the model on a GPU, the exact RAM usage depends on the model and should always be tested, but a
conservative estimate is to have **8 GB of memory** available.

The CPU performance depends on the available CPU resources, however, this should be fine-tuned for the deployment
infrastructure. By default, PyTorch will try to utilize all CPU cores to 100% and run as many threads as there are
cores. This can cause major computational overhead if the worker is deployed on large nodes. The **number of threads
used should be limited** using the `MKL_NUM_THREADS` environment variable or the `docker run` flag `--cpuset-cpus`.

Limiting CPU usage by docker configuration which only limits CPU shares is not sufficient (e.g. `docker run` flag
`--cpus` or the CPU limit in K8s, unless the non-default
[static CPU Manager policy](https://kubernetes.io/docs/tasks/administer-cluster/cpu-management-policies/) is used). For
example, on a node with 128 cores, setting the CPU limit at `16.0` results in 128 parallel threads running with each one
utilizing only 1/8 of each core's computational potential. This amplifies the effect of multithreading overhead and can
result in inference speeds up to 20x slower than expected.

Although the optimal number of threads depends on the exact model and infrastructure used, a good starting point is
around `16` (the default in the included docker image). With optimal configuration and modern hardware, the worker
should be able to process ~7 sentences per second. For more information, please refer to
[PyTorch documentation](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html).
