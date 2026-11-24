import logging
import logging.config
import os
import unittest

from nmt_worker import Translator, read_model_config
from nmt_worker.prompts import build_prompt, prepare_references, prompt_text
from nmt_worker.schemas import Response, Request, Reference

REFERENCES = [Reference(source="Keegi ei tea, kus ma olen.", target="Niemand weiß, wo ich bin."),
              Reference(source="- Muidugi mitte, ma olen pime.", target="- Nein, ich bin blind."),
              Reference(source="Me ei pea sellised olema.", target="Wir müssen nie so sein.")]


class Prompts(unittest.TestCase):
    """
    Check that the configured prompts match the trainer's examples. Does not load the model.
    """
    config = read_model_config(os.environ.get('MODEL_CONFIG', 'config/config.yaml'))

    def test_prompt(self):
        prompt = prompt_text(build_prompt(self.config, "Ma ei muretse.", "Estonian", "German"))
        self.assertIn("<|user_start|>Translate from Estonian to German\n\nMa ei muretse.<|user_end|>", prompt)

    def test_reference_prompt(self):
        prompt = prompt_text(build_prompt(self.config, "Ma ei muretse, kus ma olen või kuidas ma sinna sattusin.",
                                          "Estonian", "German", prepare_references(REFERENCES)))
        self.assertIn(
            "<|user_start|>Translate from Estonian to German, using provided relevant reference examples. Reference "
            "examples: -Example 1 Estonian: Keegi ei tea, kus ma olen. German: Niemand weiß, wo ich bin. -Example 2 "
            "Estonian: - Muidugi mitte, ma olen pime. German: - Nein, ich bin blind. -Example 3 Estonian: Me ei pea "
            "sellised olema. German: Wir müssen nie so sein. Now translate Estonian text:\n\n"
            "Ma ei muretse, kus ma olen või kuidas ma sinna sattusin.<|user_end|>", prompt)


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

    def test_references(self):
        """
        Check that a request with reference translations is translated.
        """
        request = Request(text="Ma ei muretse, kus ma olen või kuidas ma sinna sattusin.",
                          src="est",
                          tgt="ger",
                          references=REFERENCES)
        response = self.translator.process_request(request)
        self.assertIsInstance(response.result, str)
        self.assertTrue(response.result.strip())

    def test_use_references_without_references(self):
        """
        Check that a request that asks for references without giving any is translated.
        """
        request = Request(text="Aga kuidagi peab hakkama saama.",
                          src="est",
                          tgt="eng",
                          use_references=True)
        response = self.translator.process_request(request)
        self.assertTrue(response.result.strip())


if __name__ == '__main__':
    logging.config.fileConfig('config/logging.debug.ini')
    unittest.main()
