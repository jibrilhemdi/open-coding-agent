# src/data_loader.py

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def load_csv(input_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Load and validate a CSV file for the open‑coding pipeline.

    Expects `input_config` with keys:
        - file_path: path to the CSV file (str)
        - text_column: name of the column containing raw text (str)
    
    Returns a list of dictionaries, each with at least:
        - 'id': a row identifier (from an 'id' column if present, otherwise the row index)
        - 'text': the raw text content
    
    Raises:
        FileNotFoundError: if the CSV file does not exist.
        ValueError: if the text_column is missing from the CSV.
    """
    file_path = Path(input_config["file_path"])
    text_col = input_config["text_column"]

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    logger.info(f"Reading CSV from {file_path}")
    # Use dtype=str to avoid pandas auto‑conversion (e.g., numeric IDs becoming floats)
    df = pd.read_csv(file_path, dtype=str, keep_default_na=False)

    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found in CSV. Available columns: {list(df.columns)}"
        )

    # Determine ID column – use 'id' if it exists, otherwise row index
    id_column = "id" if "id" in df.columns else None

    # Log warnings for empty entries
    empty_count = df[text_col].str.strip().eq("").sum()
    if empty_count > 0:
        logger.warning(f"Found {empty_count} empty/blank entries in column '{text_col}'")

    # Convert DataFrame to list of dicts
    rows = []
    for idx, series in df.iterrows():
        row_id = series[id_column] if id_column else str(idx)
        text = series[text_col]
        rows.append({"id": row_id, "text": text})

    logger.info(f"Loaded {len(rows)} rows from {file_path}")
    return rows