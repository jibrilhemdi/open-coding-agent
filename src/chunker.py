# src/chunker.py

from abc import ABC, abstractmethod
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class Chunker(ABC):
    """Abstract base class for text chunking strategies."""

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """
        Split a raw text string into a list of smaller chunks.

        Args:
            text: The full text to be segmented.

        Returns:
            A list of chunk strings.
        """
        pass


class SentenceChunker(Chunker):
    """
    Chunk text into individual sentences using NLTK's sentence tokenizer.
    Requires the 'punkt' tokenizer model to be downloaded.
    """

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        try:
            import nltk
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                logger.info("Downloading NLTK 'punkt' tokenizer...")
                nltk.download("punkt", quiet=True)
            sentences = nltk.sent_tokenize(text)
            # Filter out any empty strings (should not happen with sent_tokenize, but safe)
            sentences = [s.strip() for s in sentences if s.strip()]
            return sentences
        except Exception as e:
            logger.error(f"Sentence chunking failed: {e}. Falling back to paragraph chunker.")
            return ParagraphChunker().chunk(text)


class ParagraphChunker(Chunker):
    """
    Chunk text into paragraphs defined by blank-line separators.
    If there are no blank lines, the whole text is treated as one paragraph.
    """

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        # Split on one or more blank lines
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            # If no double newlines, fall back to single newlines as paragraph markers
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        return paragraphs


class FixedTokenChunker(Chunker):
    """
    Chunk text into segments of a roughly fixed number of tokens/words.
    Uses whitespace splitting as a simple approximation of token count.

    Args:
        max_tokens: Maximum number of words per chunk.
    """

    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        words = text.split()
        chunks = []
        current_chunk = []
        current_count = 0

        for word in words:
            current_chunk.append(word)
            current_count += 1
            if current_count >= self.max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_count = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


def get_chunker(chunking_config: Dict[str, Any]) -> Chunker:
    """
    Factory function to instantiate the appropriate chunker based on configuration.

    Expected keys in chunking_config:
        - strategy (str): one of "sentence", "paragraph", "fixed_tokens"
        - max_tokens (int, optional): used when strategy is "fixed_tokens"

    Returns:
        An instance of a Chunker subclass.
    """
    strategy = chunking_config.get("strategy", "sentence").lower()
    if strategy == "sentence":
        return SentenceChunker()
    elif strategy == "paragraph":
        return ParagraphChunker()
    elif strategy == "fixed_tokens":
        max_tokens = chunking_config.get("max_tokens", 512)
        return FixedTokenChunker(max_tokens=max_tokens)
    else:
        logger.warning(f"Unknown chunking strategy '{strategy}'. Falling back to sentence chunker.")
        return SentenceChunker()