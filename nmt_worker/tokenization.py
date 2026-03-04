from typing import List, Tuple
from nltk import sent_tokenize


def chunk_by_tokens(
    sentences: List[str], 
    delimiters: List[str], 
    tokenizer, 
    max_tokens: int = 3000,
    prompt_overhead: int = 100
) -> List[Tuple[List[str], List[str]]]:
    """
    Group sentences into chunks that fit within the token budget.
    
    Args:
        sentences: List of sentences to chunk
        delimiters: List of delimiters between sentences
        tokenizer: HuggingFace tokenizer for counting tokens
        max_tokens: Maximum tokens per chunk (default 3000, leaving room for output)
        prompt_overhead: Estimated tokens for prompt template
    
    Returns:
        List of (sentences, delimiters) tuples for each chunk
    """
    if not sentences or sentences == ['']:
        return [([''], ['', ''])]
    
    chunks = []
    current_sentences = []
    current_delimiters = [delimiters[0]] 
    current_tokens = prompt_overhead
    
    for i, sentence in enumerate(sentences):

        sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))
        delimiter_tokens = len(tokenizer.encode(delimiters[i + 1], add_special_tokens=False)) if i + 1 < len(delimiters) else 0
        total_new_tokens = sentence_tokens + delimiter_tokens
        
        if current_tokens + total_new_tokens > max_tokens and current_sentences:
            # Save current chunk and start a new one
            # Include the trailing delimiter for the last sentence
            current_delimiters.append(delimiters[i] if i < len(delimiters) else '')
            chunks.append((current_sentences, current_delimiters))
            
            # Start new chunk
            current_sentences = [sentence]
            current_delimiters = ['']  # No leading delimiter for continuation
            current_tokens = prompt_overhead + sentence_tokens
        else:
            # Add sentence to current chunk
            current_sentences.append(sentence)
            current_delimiters.append(delimiters[i + 1] if i + 1 < len(delimiters) else '')
            current_tokens += total_new_tokens
    
    # the last chunk
    if current_sentences:
        chunks.append((current_sentences, current_delimiters))
    
    return chunks


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
