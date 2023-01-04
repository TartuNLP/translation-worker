import unittest

from nmt_worker import Translator, read_model_config
from nmt_worker.schemas import Response, Request


class Septilang(unittest.TestCase):
    translator: Translator
    config = 'config/config.yaml'
    model = 'septilang'

    @classmethod
    def setUpClass(cls):
        model_config = read_model_config(cls.config, cls.model)
        cls.translator = Translator(model_config)

    def test_text_translation(self):
        """
        Check that a response object is returned upon text translation request.
        """
        request = Request(text="Tere! Teretulemast!",
                          src="est",
                          tgt="eng")
        response = self.translator.process_request(request)
        self.assertIsInstance(response, Response)
        self.assertIsInstance(response.result, str)

    def test_list_translation(self):
        """
        Check that lists are translated appropriately.
        """
        request = Request(text=["Tere!", "Teretulemast!"],
                          src="est",
                          tgt="eng")
        response = self.translator.process_request(request)
        self.assertIsInstance(response, Response)
        self.assertIsInstance(response.result, list)
        self.assertEqual(len(response.result), len(request.text))


if __name__ == '__main__':
    unittest.main()
