# src/utils.py

import logging
import logging.config
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def setup_logging(logging_config: Dict[str, Any]) -> None:
    """
    Configure application logging based on the provided configuration dictionary.

    Expected keys in logging_config:
        - level (str): logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        - file (str, optional): path to a log file; if omitted, logs to console only.

    The format includes timestamp, logger name, level, and message.
    """
    level = logging_config.get("level", "INFO").upper()
    log_file = logging_config.get("file")

    handlers: list[logging.Handler] = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)

    # File handler (if log file specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Build a new filename with timestamp: pipeline_20260123_153045.log
        stem = log_path.stem           # e.g. "pipeline"
        suffix = log_path.suffix       # e.g. ".log"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"{stem}_{timestamp}{suffix}"
        new_path = log_path.parent / new_name

        file_handler = logging.FileHandler(new_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_formatter)
        handlers.append(file_handler)

        # Optional: log which file is being used
        print(f"Logging to file: {new_path}")

    # Configure root logger (or you can configure a named logger for the package)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # Remove any default handlers to avoid duplication if called multiple times
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    for h in handlers:
        root_logger.addHandler(h)

    # Optionally, suppress overly verbose loggers from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def merge_config_with_env(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge environment variables into the configuration dictionary.

    This function looks for:
    - OLLAMA_HOST  -> sets config["coding"]["ollama_host"] if present
    - OLLAMA_MODEL -> sets config["coding"]["model"] if present (overrides YAML)

    The function also loads a .env file if present in the current directory.

    Returns a (potentially) new config dict (does not modify the original).
    """
    # Load .env file if present (uses python-dotenv)
    try:
        from dotenv import load_dotenv
        load_dotenv()  # loads .env from current working directory
    except ImportError:
        pass

    import copy
    config = copy.deepcopy(config)  # avoid side effects

    # Ensure coding section exists
    if "coding" not in config:
        config["coding"] = {}

    host = os.getenv("OLLAMA_HOST")
    if host:
        config["coding"]["ollama_host"] = host

    model = os.getenv("OLLAMA_MODEL")
    if model:
        config["coding"]["model"] = model

    return config


def load_prompt_template(template_path: str) -> str:
    """
    Load a prompt template from a file.

    Args:
        template_path: path to the prompt template text file.

    Returns:
        The content of the prompt template as a string.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    path = Path(template_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()