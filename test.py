import unittest

from nmt_worker import Translator, read_model_config
from nmt_worker.schemas import Response, Request


class NLLBBasedFromEst(unittest.TestCase):
    translator: Translator
    config = 'config/config.yaml'
    model = 'nllb_based_from_est'

    @classmethod
    def setUpClass(cls):
        model_config = read_model_config(cls.config, cls.model)
        cls.translator = Translator(model_config)

    def test_text_translation_est_eng(self):
        """
        Check that a response object is returned upon text translation request.
        """
        request = Request(text="Tere! Teretulemast!",
                          src="est",
                          tgt="eng")
        response = self.translator.process_request(request)
        self.assertIsInstance(response, Response)
        self.assertIsInstance(response.result, str)

    def test_text_translation_est_ger(self):
        """
        Check that a response object is returned upon text translation request.
        """
        request = Request(text="Tere! Teretulemast!",
                          src="est",
                          tgt="ger")
        response = self.translator.process_request(request)
        self.assertIsInstance(response, Response)
        self.assertIsInstance(response.result, str)

    def test_list_translation_est_eng(self):
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

    def test_text_translation_est_rus(self):
        """
        Check that a response object is returned upon text translation request.
        """
        request = Request(text="Tere! Teretulemast!",
                          src="est",
                          tgt="rus")
        response = self.translator.process_request(request)
        self.assertIsInstance(response, Response)
        self.assertIsInstance(response.result, str)

    def test_text_translation_est_ukr(self):
        """
        Check that a response object is returned upon text translation request.
        """
        request = Request(text="Tere! Teretulemast!",
                          src="est",
                          tgt="ukr")
        response = self.translator.process_request(request)
        self.assertIsInstance(response, Response)
        self.assertIsInstance(response.result, str)


class NLLBBasedIntoEst(unittest.TestCase):
    translator: Translator
    config = 'config/config.yaml'
    model = 'nllb_based_into_est'

    @classmethod
    def setUpClass(cls):
        model_config = read_model_config(cls.config, cls.model)
        cls.translator = Translator(model_config)

    def test_text_translation_eng_est(self):
        """
        Check that a response object is returned upon text translation request.
        """
        request = Request(text="Hello, it's nice to meet you!",
                          src="eng",
                          tgt="est")
        response = self.translator.process_request(request)
        self.assertIsInstance(response, Response)
        self.assertIsInstance(response.result, str)

    def test_text_translation_rus_est(self):
        """
        Check that a response object is returned upon text translation request.
        """
        request = Request(text="На улице дождь или солнечно?",
                          src="rus",
                          tgt="est")
        response = self.translator.process_request(request)
        self.assertIsInstance(response, Response)
        self.assertIsInstance(response.result, str)

    def test_text_translation_ger_est(self):
        """
        Check that a response object is returned upon text translation request.
        """
        request = Request(text="Ich lebe schon seit sechs Jahren hier.",
                          src="ger",
                          tgt="est")
        response = self.translator.process_request(request)
        self.assertIsInstance(response, Response)
        self.assertIsInstance(response.result, str)


if __name__ == '__main__':
    unittest.main()
