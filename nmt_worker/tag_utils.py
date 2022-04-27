"""
Fallback preprocessing method to meet the bare minimum of tagged translation constraints.
All tags are removed and appended to the translation.
"""
import re
import html
from typing import List, Tuple

from nmt_worker.schemas import InputType

tag_patterns = {
    InputType.SDL: r'<[0-9]+ id=[0-9]+/?>|</[0-9]+>',
    InputType.MEMOQ: r'<[^>]*>'
}

bpt = r'<[^/>]*>'
ept = r'</[^>]*>'

# Other symbols do not need replacing
html_entities = {'<': '&lt;',
                 '>': '&gt;',
                 '&': '&amp;'}


def preprocess_tags(sentences: List[str], input_type: InputType) -> (List[str], List[List[Tuple[str, int, str]]]):
    if input_type in tag_patterns:
        pattern = tag_patterns[input_type]
        clean_sentences = []
        tags = []
        for sentence in sentences:
            sentence = sentence.strip()
            sentence_tags = []  # list of tuples (tag, indexes, tag_type)

            tokens = list(filter(None, re.split(rf' |{pattern}', sentence)))
            tokens_w_tags = list(filter(None, re.split(rf' |({pattern})', sentence)))

            clean_sentences.append(' '.join(tokens).strip())

            for idx, item in enumerate(tokens_w_tags):
                idx = idx - len(sentence_tags)
                if len(tokens) <= idx or item != tokens[idx]:
                    if len(tokens) <= idx:
                        idx = -1

                    if re.match(bpt, item):
                        sentence_tags.append((item, idx, 'bpt'))
                    elif re.match(ept, item):
                        sentence_tags.append((item, idx, 'ept'))
                    else:
                        sentence_tags.append((item, idx, 'ph'))

            tags.append(sentence_tags)

    else:
        clean_sentences = sentences
        tags = [[] for _ in sentences]

    clean_sentences = [html.unescape(sentence) for sentence in clean_sentences]

    return clean_sentences, tags


def postprocess_tags(translations: List[str], tags: List[List[Tuple[str, int, str]]], input_type: InputType):
    translations = [sentence.replace("<unk>", "") for sentence in translations]

    if input_type in tag_patterns:
        for symbol, entity in html_entities.items():
            translations = [sentence.replace(symbol, entity) for sentence in translations]

    retagged = []

    for translation, sentence_tags in zip(translations, tags):
        retagged_sentence = []

        tokens = translation.split(' ')

        for idx, token in enumerate(tokens):
            whitespace_added = False
            while sentence_tags and sentence_tags[0][1] == idx:
                if not whitespace_added and sentence_tags[0][2] == 'bpt':
                    retagged_sentence.append(' ')
                    whitespace_added = True
                retagged_sentence.append(sentence_tags.pop(0)[0])
            if not whitespace_added:
                retagged_sentence.append(' ')
            retagged_sentence.append(token)

        retagged.append((''.join(retagged_sentence) + ''.join([tag for tag, _, _ in sentence_tags])).strip())

    return retagged
