import logging
import logging.config
import os
import unittest

from nmt_worker import Translator, read_model_config
from nmt_worker.schemas import Response, Request


class Apertus(unittest.TestCase):
    translator: Translator
    config = os.environ.get('MODEL_CONFIG', 'config/config.yaml')

    @classmethod
    def setUpClass(cls):
        model_config = read_model_config(cls.config)
        cls.translator = Translator(model_config)

    def test_text_translation(self):
        """
        Check that a response object is returned upon text translation request.
        """
        request = Request(text="Aga kuidagi peab hakkama saama.",
                          src="est",
                          tgt="eng")
        response = self.translator.process_request(request)
        self.assertIsInstance(response, Response)
        self.assertIsInstance(response.result, str)
        self.assertTrue(response.result.strip())

    def test_list_translation(self):
        """
        Check that lists are translated appropriately.
        """
        request = Request(text=["Tere!", "", "Magasin end välja ja olin eluga rahul."],
                          src="est",
                          tgt="eng")
        response = self.translator.process_request(request)
        self.assertIsInstance(response, Response)
        self.assertIsInstance(response.result, list)
        self.assertEqual(len(response.result), len(request.text))
        self.assertEqual(response.result[1], '')

    def test_structure(self):
        """
        Check that the sentence and paragraph structure of the input is kept.
        """
        request = Request(text="Täna on ilus ilm. Lähen jalutama.\n\nHomme sajab vihma.",
                          src="est",
                          tgt="eng")
        response = self.translator.process_request(request)
        paragraphs = response.result.split('\n\n')
        self.assertEqual(len(paragraphs), 2)
        self.assertTrue(all(paragraph.strip() for paragraph in paragraphs))


if __name__ == '__main__':
    logging.config.fileConfig('config/logging.debug.ini')
    unittest.main()
