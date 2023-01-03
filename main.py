import logging.config
from argparse import ArgumentParser, FileType

from nmt_worker import MQConsumer, Translator, MQConfig, read_model_config


def parse_args():
    parser = ArgumentParser(
        description="A neural machine translation worker that processes incoming translation requests via RabbitMQ."
    )
    parser.add_argument('--model-config', type=FileType('r'), default='config/config.yaml',
                        help="The model config YAML file to load.")
    parser.add_argument('--model-name', type=str,
                        help="The model to load. Refers to the model name in the config file.")
    parser.add_argument('--log-config', type=FileType('r'), default='config/logging.prod.ini',
                        help="Path to log config file.")

    return parser.parse_args()


def main():
    args = parse_args()
    logging.config.fileConfig(args.log_config.name)
    model_config = read_model_config(args.model_config.name, args.model_name)
    mq_config = MQConfig()

    translator = Translator(model_config)
    consumer = MQConsumer(
        translator=translator,
        mq_config=mq_config
    )

    consumer.start()


if __name__ == "__main__":
    main()
