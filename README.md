# TartuNLP translation service
Service to run our modularized multilingual translation models.
 
The service is designed to connect to a RabbitMQ server to process translation requests from a queue. We have also
 included a demo Flask application that can be run standalone.

### Requirements
The following steps are required to install all prerequisites in a CPU Conda environment.

```
conda env create -f environment.yml -n nmt
conda activate nmt
python -c "import nltk; nltk.download(\"punkt\"); nltk.download(\"cmudict\")"
```

### Configuration and model files
Sample configurations are included in the config directory for both CPU and GPU usage.

When running the Flask application, any RabbitMQ parameters are ignored. When running the RabbitMQ version
, connection parameters should be defined as environment variables or in `config/.env`. The required connection
 variable names can be seen in `config/sample.env`.
  
### Usage
The CPU-based Flask application can be tested with the command below.

```
python app.py --config config/modular.cpu.ini
```
This will start a CPU-based Werkzeug development server where translation functionality is available via a POST request:
```
POST 127.0.0.1:5000/translation
{
     "text": "Tere",
     "src": "et",
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
 
To run the RabbitMQ-based service, run `nmt_service.py` instead of `app.py`. Running this service is also possible using the included Dockerfile.
  

