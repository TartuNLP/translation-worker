from typing import List
from nltk import sent_tokenize


def sentence_tokenize(text: str) -> tuple[List, List]:
    """
    Split text for sentence-by-sentence translation model.
    """
    if not text.strip():
        return [''], ['']
        
    lines = text.split('\n')
    
    sentences = []
    delimiters = ['']
    
    for line_idx, line in enumerate(lines):
        line = line.strip()
        
        if not line: 
            if sentences:
                # newline to the last delimiter
                delimiters[-1] += '\n'
            continue
            
        # Check if this line looks like a standalone header/title
        is_standalone = (
            len(line) < 100 and  # Relatively short
            not line.endswith(('.', '!', '?', '"', "'", ':', ';')) and # No typical sentence ending
            not line.startswith('"') and  # Not a quote continuation
            '\n' not in line  # Single line
        )
        
        if is_standalone:
            # Treat standalone lines as separate sentences
            sentences.append(line)
            delimiters.append('\n' if line_idx < len(lines) - 1 else '')
        else:
            # Use NLTK to split this line into sentences
            line_sentences = [sent.strip() for sent in sent_tokenize(line) if sent.strip()]
            
            for sent_idx, sentence in enumerate(line_sentences):
                sentences.append(sentence)
                
                # Determine delimiter for this sentence
                if sent_idx == len(line_sentences) - 1:  # Last sentence in this line
                    if line_idx < len(lines) - 1:  # Not the last line
                        delimiters.append('\n')
                    else:  # Last line
                        delimiters.append('')
                else:  # Not the last sentence in this line
                    delimiters.append(' ')
    
    if len(sentences) == 0:
        return [''], ['']
    
    while len(delimiters) <= len(sentences):
        delimiters.append('')
    
    delimiters = delimiters[:len(sentences) + 1]
    
    return sentences, delimiters
