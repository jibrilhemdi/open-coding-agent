# src/filter.py
"""
Lightweight keyword‑based filter to skip chunks that are likely off‑topic.
Keywords are loaded from a configurable source (YAML, text file, or a built‑in list).
"""

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# A small default list – you’ll typically override this via config
DEFAULT_KEYWORDS = [
    "roleplay", "role-play", "rp", "larp",
    "character", "acting", "plot", "pretend", "erp",
    "ai rp", "c.ai", "bot role", "character ai",
    "gemini", "claude", "openai", "janitor ai"
]


class KeywordFilter:
    """
    A simple filter that checks if any of the configured keywords appear in a text.
    Keywords are case‑insensitive and can be loaded from a list or a file.
    """

    def __init__(self, keywords: Optional[List[str]] = None):
        """
        Args:
            keywords: A list of keyword strings. If None, uses DEFAULT_KEYWORDS.
        """
        self.keywords = [k.lower().strip() for k in (keywords or DEFAULT_KEYWORDS)]
        logger.info(f"Filter initialised with {len(self.keywords)} keywords")

    def is_relevant(self, text: str) -> bool:
        """Return True if the text contains at least one keyword."""
        lower_text = text.lower()
        return any(kw in lower_text for kw in self.keywords)

    @classmethod
    def from_config(cls, filter_config: dict) -> "KeywordFilter":
        """
        Instantiate the filter from a configuration dictionary.
        Expected keys:
            keywords: list of keyword strings (optional)
            keywords_file: path to a text file with one keyword per line (optional)
        If both are provided, they are merged.
        """
        keywords = filter_config.get("keywords", [])
        file_path = filter_config.get("keywords_file")

        if file_path:
            path = Path(file_path)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    file_kws = [line.strip() for line in f if line.strip()]
                keywords += file_kws
            else:
                logger.warning(f"Keywords file not found: {path}")

        if not keywords:
            keywords = DEFAULT_KEYWORDS
            logger.info("No custom keywords provided, using defaults")

        return cls(keywords=keywords)