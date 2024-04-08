import logging
import unittest

from nmt_worker import Translator, read_model_config
from nmt_worker.schemas import Response, Request


class Smugri3(unittest.TestCase):
    translator: Translator
    config = 'config/config.yaml'
    model = 'smugri3'

    @classmethod
    def setUpClass(cls):
        model_config = read_model_config(cls.config, cls.model)
        cls.translator = Translator(model_config)

    def test_text_translation(self):
        """
        Check that a response object is returned upon text translation request.
        """
        request = Request(text="Aga kuidagi peab hakkama saama.",
                          src="est",
                          tgt="vro")
        response = self.translator.process_request(request)
        self.assertIsInstance(response, Response)
        self.assertIsInstance(response.result, str)

    def test_list_translation(self):
        """
        Check that lists are translated appropriately.
        """
        request = Request(text=["Tere!", "Tere tulemast!", "Magasin end välja ja olin eluga rahul."],
                          src="est",
                          tgt="vro")
        response = self.translator.process_request(request)
        self.assertIsInstance(response, Response)
        self.assertIsInstance(response.result, list)
        self.assertEqual(len(response.result), len(request.text))


class Smugri3_14(unittest.TestCase):
    translator: Translator
    config = 'config/config.yaml'
    model = 'smugri3_14'

    @classmethod
    def setUpClass(cls):
        model_config = read_model_config(cls.config, cls.model)
        cls.translator = Translator(model_config)

    def test_text_translation(self):
        """
        Check that a response object is returned upon text translation request.
        """
        request = Request(text="Aga kuidagi peab hakkama saama.",
                          src="est",
                          tgt="vro")
        response = self.translator.process_request(request)
        self.assertIsInstance(response, Response)
        self.assertIsInstance(response.result, str)

    def test_list_translation(self):
        """
        Check that lists are translated appropriately.
        """
        request = Request(text=["Tere!", "Tere tulemast!", "Magasin end välja ja olin eluga rahul."],
                          src="est",
                          tgt="vro")
        response = self.translator.process_request(request)
        self.assertIsInstance(response, Response)
        self.assertIsInstance(response.result, list)
        self.assertEqual(len(response.result), len(request.text))


if __name__ == '__main__':
    logging.config.fileConfig('config/logging.debug.ini')
    unittest.main()
