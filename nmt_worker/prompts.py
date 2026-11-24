"""
Building the translation prompts from the templates in the model config.

A prompt is built as a list of (string, is_request_text) parts. The strings joined together give the prompt; the parts
that come from the request (the input text and the references) are marked, so that special token strings in them can
be tokenized as plain text.
"""
from string import Formatter
from typing import List, Optional, Tuple, Union

from .config import ModelConfig
from .normalization import normalize
from .schemas import Reference

PromptParts = List[Tuple[str, bool]]


def fill_template(template: str, **values: Union[str, PromptParts]) -> PromptParts:
    """
    Fill the placeholders of a str.format() style template. A str value is trusted text (e.g. a language name from the
    config), a list of parts is inserted as it is, so request text is passed as [(text, True)].
    """
    parts = []
    for literal, name, _, _ in Formatter().parse(template):
        if literal:
            parts.append((literal, False))
        if name is not None:
            value = values[name]
            parts.extend([(value, False)] if isinstance(value, str) else value)
    return parts


def prompt_text(parts: PromptParts) -> str:
    return ''.join(part for part, _ in parts)


def prepare_references(references: List[Reference]) -> List[Reference]:
    """
    Put each side of a reference on one line, as the examples are separated by spaces only in the prompt, and normalize
    it the same way as the input text. References with an empty side are dropped.
    """
    prepared = []
    for reference in references:
        source, target = (normalize(' '.join(side.split())).strip() for side in (reference.source, reference.target))
        if source and target:
            prepared.append(Reference(source=source, target=target))
    return prepared


def reference_example(config: ModelConfig, number: int, reference: Reference, src_lang: str,
                      tgt_lang: str) -> PromptParts:
    return fill_template(config.reference_example_template, number=str(number), src_lang=src_lang, tgt_lang=tgt_lang,
                         source=[(reference.source, True)], target=[(reference.target, True)])


def build_prompt(config: ModelConfig, text: str, src_lang: str, tgt_lang: str,
                 references: Optional[List[Reference]] = None) -> PromptParts:
    """
    The prompt for translating one segment, with the reference examples numbered in the given order if there are any.
    """
    if not references:
        return fill_template(config.prompt_template, src_lang=src_lang, tgt_lang=tgt_lang, text=[(text, True)])
    examples = [part for number, reference in enumerate(references, 1)
                for part in reference_example(config, number, reference, src_lang, tgt_lang)]
    return fill_template(config.reference_prompt_template, src_lang=src_lang, tgt_lang=tgt_lang, examples=examples,
                         text=[(text, True)])
