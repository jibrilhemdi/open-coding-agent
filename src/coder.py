# src/coder.py

import logging
import time
import ollama
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Fallback prompt when no external template is provided
DEFAULT_PROMPT_TEMPLATE = (
    "You are a qualitative researcher performing open coding. "
    "Read the following text and suggest 1-3 concise thematic codes that capture its meaning. "
    "Output only the codes, separated by commas.\n\n"
    "Text: {text}"
)


class OpenCoder:
    """
    Handles communication with a local Ollama LLM to perform open coding on text chunks.
    """

    def __init__(self, coding_config: Dict[str, Any]):
        """
        Args:
            coding_config: dict with keys:
                - model (str): Ollama model name (e.g. 'llama3.2')
                - prompt_template_path (str, optional): path to a custom prompt file
                - temperature (float, optional): model temperature (default 0.0)
                - max_retries (int, optional): number of retries on failure (default 3)
        """
        self.model = coding_config["model"]
        self.temperature = coding_config.get("temperature", 0.0)
        self.max_retries = coding_config.get("max_retries", 3)

        # Load prompt template
        prompt_template = coding_config.get("prompt_template")
        if prompt_template:
            self.prompt_template = prompt_template
        else:
            template_path = coding_config.get("prompt_template_path")
            if template_path and Path(template_path).exists():
                with open(template_path, "r") as f:
                    self.prompt_template = f.read()
                logger.info(f"Loaded prompt template from {template_path}")
            else:
                if template_path:
                    logger.warning(
                        f"Prompt template file not found: {template_path}. Using default."
                    )
                self.prompt_template = DEFAULT_PROMPT_TEMPLATE

    def code(self, text: str) -> str:
        """
        Send a chunk of text to the LLM and return a comma‑separated string of codes.

        Args:
            text: The chunk of text to be coded.

        Returns:
            A string of codes (e.g. "code1, code2") or an empty string on failure.
        """
        if not text or not text.strip():
            return ""

        prompt = self.prompt_template.format(text=text)

        for attempt in range(self.max_retries):
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": self.temperature},
                )
                content = response["message"]["content"].strip()

                if any(
                    content.lower().startswith(prefix)
                    for prefix in ("please provide the text", "please provide")
                ):
                    raise ValueError("LLM returned a placeholder instead of codes")

                if not content:
                    raise ValueError("Empty response from the model")

                logger.debug(f"LLM response: {content}")
                codes = self._clean_response(content)
                return codes

            except Exception as e:
                logger.error(
                    f"LLM call failed (attempt {attempt+1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    sleep_time = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                    time.sleep(sleep_time)
                else:
                    logger.error("Max retries exceeded. Returning empty codes.")
                    return ""

        return ""

    @staticmethod
    def _clean_response(response: str) -> str:
        """
        Post‑process the LLM output to extract clean, comma‑separated codes.

        Handles common formatting pitfalls: prefixes like 'Codes:', bullet points,
        numbered lists, quotes, and line breaks.
        """
        import re

        # Remove leading "Codes:" (case‑insensitive)
        cleaned = re.sub(r"(?i)^\s*codes?\s*:\s*", "", response)
        # Strip surrounding quotes
        cleaned = cleaned.strip("\"'")
        # Remove bullet points or numbering like "1. ", " - "
        cleaned = re.sub(r"(^\d+\.\s*|\s*-\s*)", "", cleaned, flags=re.MULTILINE)
        # Replace newlines with commas
        cleaned = cleaned.replace("\n", ",")
        # Split, strip whitespace, and filter out empty strings
        codes = [c.strip() for c in cleaned.split(",") if c.strip()]
        return ", ".join(codes)