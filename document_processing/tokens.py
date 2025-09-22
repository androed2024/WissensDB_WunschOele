"""
Token counting utility for chunking decisions.
Uses tiktoken for accurate token counting with fallback to word estimation.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

def count_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    """
    Count tokens in text using tiktoken.
    Falls back to word count * 1.3 if tiktoken fails.
    
    Args:
        text: Text to count tokens for
        model: Model name to use for tokenizer (default: text-embedding-3-small)
        
    Returns:
        Approximate token count
    """
    if not text or not text.strip():
        return 0
        
    try:
        import tiktoken
        
        # Get encoding for the model
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base encoding if model not found
            encoding = tiktoken.get_encoding("cl100k_base")
            logger.debug(f"Model '{model}' not found, using cl100k_base encoding")
        
        tokens = encoding.encode(text)
        return len(tokens)
        
    except ImportError:
        logger.warning("tiktoken not available, falling back to word count estimation")
        word_count = len(text.split())
        return int(word_count * 1.3)
        
    except Exception as e:
        logger.warning(f"Error counting tokens with tiktoken: {e}, falling back to word count")
        word_count = len(text.split())
        return int(word_count * 1.3)


def estimate_tokens_from_chars(char_count: int) -> int:
    """
    Quick estimation of token count from character count.
    Rule of thumb: ~4 characters per token for German/English text.
    
    Args:
        char_count: Number of characters
        
    Returns:
        Estimated token count
    """
    return max(1, char_count // 4)


def estimate_chars_from_tokens(token_count: int) -> int:
    """
    Quick estimation of character count from token count.
    Rule of thumb: ~4 characters per token for German/English text.
    
    Args:
        token_count: Number of tokens
        
    Returns:
        Estimated character count
    """
    return token_count * 4
