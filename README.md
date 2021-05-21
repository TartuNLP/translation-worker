# TartuNLP translation service
Service to run our multilingual multi-domain neural machine translation model. The current model supports seven
 languages as input and output: Estonian, Latvian, Lithuanian, English, Russian, German and Finnish. The
  model has been trained with Sockeye and uses the MXNet deep learning library.
 
The service is designed to connect to a RabbitMQ server to process translation requests from a queue. We have also
 included a demo Flask application that can be run standalone.

### Requirements
The repository contains a submodule for truecasing, 
therefore use the `--recurse-submodules` flag during cloning or `git submodule update --init` after cloning.

The following steps are required to install all prerequisites in a CPU Conda environment. The code has been tested
 with Ubuntu 18.04 and 20.04.
```
conda env create -f environment.yml -n nmt
conda activate nmt
pip install -e nauron/
python -c "import nltk; nltk.download(\"punkt\"); nltk.download(\"cmudict\")"
pip install sockeye==1.18.106 --no-deps
```

In a GPU-based installation, MXNet should be removed from the environment.yml file, and an MXNet verion with CUDA
 support should be installed separately. For this, use the guide at https://mxnet.apache.or/versions/1.8.0/get_started to find the correct version. Please note that the models used in this repository will not work with
  newer Sockeye versions.

### Configuration and model files
We have included sample configuration files for our public model (septilang) and our Finno-Ugric model
 (smugri) in the `config/` directory for both CPU and GPU usage. The model and logfile locations may need to be
  modified depending on your setup.
  
The public model can be downloaded from the "Releases" section. To use the Finno-Ugric model, please contact us.
  
When running the Flask application, any RabbitMQ parameters are ignored. When running the RabbitMQ version
, connection parameters should be defined as environment variables or in `config/.env`. The required connection
 variable names can be seen in `config/sample.env`.
  
### Usage
The CPU-based Flask application can be tested with the command below. Threading has been turned off, as MXNet is
 not thread-safe.
```
python app.py --config config/septilang.cpu.ini
```
This will start a CPU-based Werkzeug development server where translation functionality is available via a POST request:
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
 
To run the RabbitMQ-based service, run `nmt_worker.py` instead of `app.py`. Running this service is also possible using the included Dockerfile.
  

