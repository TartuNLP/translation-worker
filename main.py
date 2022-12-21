import logging.config
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter


def parse_args():
    parser = ArgumentParser(
        description="A neural machine translation engine. This application supports two modes of operation: "
                    "a) a worker that processes incoming translation requests via RabbitMQ; "
                    "b) translation of a text file into a new file;",
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model-name', type=str, required=True,
                        help="The model to load. Refers to the model name in the config file.")

    parser.add_argument('--model-config', type=FileType('r'), default='config/config.yaml',
                        help="The model config YAML file to load.")
    parser.add_argument('--log-config', type=FileType('r'), default='config/logging.prod.ini',
                        help="Path to log config file.")

    file_args = parser.add_argument_group("Local file translation arguments, if the following arguments "
                                          "exist, local file translation is started. Otherwise a RabbitMQ worker is "
                                          "started")
    file_args.add_argument('--input-file', type=FileType('r'),
                           help="Path to the input text file.")
    file_args.add_argument('--output-file', type=FileType('w'),
                           help="Path to the output text file.")
    file_args.add_argument('--input-lang', type=str,
                           help="Input language code")
    file_args.add_argument('--output-lang', type=str,
                           help="Output language code.")

    # TODO: add domain and input_type?
    # TODO: add stdin / stdout?

    return parser.parse_args()


def main():
    from nmt_worker import Translator, read_model_config

    args = parse_args()

    logging.config.fileConfig(args.log_config.name)
    model_config = read_model_config(args.model_config.name, args.model_name)

    translator = Translator(model_config)

    if args.input_file or args.output_file:
        assert args.input_file and args.output_file and args.input_lang and \
               args.output_lang, "Both input and output files must be defined."
        from nmt_worker import Request

        input_text = args.input_file.read()
        request = Request(
            text=input_text,
            src=args.input_lang,
            tgt=args.output_lang
        )
        response = translator.process_request(request)
        args.output_file.write(response.result)
    else:
        from nmt_worker import MQConsumer, MQConfig
        mq_config = MQConfig()
        consumer = MQConsumer(
            translator=translator,
            mq_config=mq_config
        )

        consumer.start()


if __name__ == "__main__":
    main()
