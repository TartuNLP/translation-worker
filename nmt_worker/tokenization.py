from typing import List
from nltk import sent_tokenize


def sentence_tokenize(text: str) -> (List, List):
    """
    Split text into sentences and save info about delimiters between them to restore linebreaks,
    whitespaces, etc.
    """
    delimiters = []
    sentences = [sent.strip() for sent in sent_tokenize(text)]
    if len(sentences) == 0:
        return [''], ['']
    else:
        try:
            for sentence in sentences:
                idx = text.index(sentence)
                delimiters.append(text[:idx])
                text = text[idx + len(sentence):]
            delimiters.append(text)
        except ValueError:
            delimiters = ['', *[' ' for _ in range(len(sentences) - 1)], '']

    return sentences, delimiters
