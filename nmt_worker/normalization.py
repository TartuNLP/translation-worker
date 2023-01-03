"""
Adapted from the Moses punctuation normalization script
"""
import re

# define and compile all the patterns
REGEXES = (
    # whitespace character normalization
    (re.compile(r' '), ' '),
    (re.compile(r'\r'), r''),

    # parenthesis
    (re.compile(r' *\( *'), r' ('),
    (re.compile(r' *\) *'), r') '),
    (re.compile(r'\) ([.!:?;,])'), r')\1'),

    # remove unnnecessary spaces
    (re.compile(r'(\d) %'), r'\1%'),
    (re.compile(r' ([:;?!])'), r'\1'),

    # normalize quotation marks, apostrophes and hyphens
    (re.compile(r'[`´′‘‚’]'), r"'"),
    (re.compile(r"''"), r'"'),
    (re.compile(r'[„“”«»]'), r'"'),
    (re.compile(r'­'), r''),
    (re.compile(r'[–‐‒−]'), r'-'),
    (re.compile(r' *— *'), r' - '),

    # various non-unicode characters
    (re.compile(r'…'), r'...'),

    # remove extra spaces
    (re.compile(r' +'), r' ')
)


def normalize(sentence: str):
    for regex, sub in REGEXES:
        sentence = regex.sub(sub, sentence)
    return sentence
