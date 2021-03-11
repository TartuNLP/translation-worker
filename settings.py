import logging.config
from os import environ
from argparse import ArgumentParser, FileType
from configparser import ConfigParser

from dotenv import load_dotenv

def load_config() -> ConfigParser:
    parser = ArgumentParser(
        description="Backend NMT server for Sockeye models."
    )

    parser.add_argument('--config-file', type=FileType('r'), default='config/config.ini', help="Path to config file.")
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config_file.name)
    return config

_config = load_config()
logging.config.fileConfig('config/logging.conf', defaults={'logfile': _config['general']['logfile']})

load_dotenv("config/.env")

CPU = eval(_config['general']['cpu'])
CHAR_LIMIT = eval(_config['general']['char_limit'])

MQ_HOST = environ.get('MQ_HOST')
MQ_PORT = environ.get('MQ_PORT')
MQ_USERNAME = environ.get('MQ_USERNAME')
MQ_PASSWORD = environ.get('MQ_PASSWORD')

MQ_EXCHANGE = _config['rabbitmq']['exchange']
MQ_QUEUE_NAME = _config['rabbitmq']['queue_name']
MQ_ALT_ROUTES = eval(_config['rabbitmq']['alt_routes'])

NMT_MODEL = _config['models']['nmt']
SPM_MODEL = _config['models']['spm']
TC_MODEL = _config['models']['tc']

FACTOR_SEQUENCE = eval(_config['factors']['sequence'])
FACTORS = eval(_config['factors']['factors'])
ALT_FACTORS = eval(_config['factors']['alt_factors'])