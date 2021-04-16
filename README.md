# TartuNLP translation service

Service to run our multilingual multi-domain neural machine translation model. The current model supports seven
 languages as input and output language: Estonian, Latvian, Lithuanian, English, Russian, German and Finnish.
 
### Usage

This service is designed to connect to a RabbitMQ server to process requests, but also contains a simple Flask
 version to run it locally with:
```
python app.py --config config/septilang.ini
```

This will run a Werkzeug development server where translation functionality is available via a POST request:
```
POST 127.0.0.1:5000/translation
{
     "text": "Tere",
     "tgt": "en",
     "domain": "auto"
}
```
Response:
```
{
    "result": "Hi."
}
```

In case the text field contains a string, it is automatically split into sentences. In case it contains a list, the
 service assumes this list to be a list of sentences and will not do any further splitting.
 
To run the RabbitMQ-based service, run `nmt_service.py` instead of `app.py` and define message queue variables in
 `config/.env` or as environment variables. Running this service is also possible using the included Dockerfile.
  
### Requirements
The repository contains a submodule used for truecasing, therefore use the `--recurse-submodules` flag during cloning
 or use `git submodule update --init` after cloning.

The following steps are required to install all prerequisites in a CPU environment. In a GPU-based installation
, the environment.yml file should be modified to use an MXNet verion with CUDA support. For this, use the guide at
 https://mxnet.apache.org/versions/1.8.0/get_started to find the correct version. Please note that the
 models used in this repository will not work with newer Sockeye versions.
```
conda env create -f environment.yml -n nmt
git clone https://github.com/TartuNLP/nauron.git && pip install -e nauron/
python -c "import nltk; nltk.download(\"punkt\"); nltk.download(\"cmudict\")"
pip install sockeye==1.18.106 --no-deps
```

We have included default configuration files for our public model (septilang) and our Finno-Ugric model
 (smugri). Both CPU and GPU configurations can be found. The public model can be downloaded from the
  "Releases" section. To use the Finno-Ugric model, please contact us.