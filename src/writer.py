# src/writer.py

import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def save_coded(results: List[Dict[str, Any]], output_config: Dict[str, Any]) -> None:
    """
    Write coded results to a CSV file.

    Args:
        results: List of dictionaries, each containing at least:
                 - id
                 - original_text
                 - chunk
                 - codes
        output_config: dict with keys:
            - file_path (str): path to the output CSV file
            - columns (list of str, optional): specific columns to include in the output.
              If not provided, uses all keys from the first result record.

    Raises:
        ValueError: if results is empty.
    """
    if not results:
        logger.warning("No results to save. Exiting writer.")
        return

    file_path = Path(output_config["file_path"])
    # Ensure output directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine which columns to write
    columns = output_config.get("columns")
    if not columns:
        columns = list(results[0].keys())  # preserve insertion order
        logger.info(f"No columns specified; using all keys: {columns}")

    # Filter records to include only desired columns (drop missing ones)
    filtered_results = []
    for record in results:
        filtered = {col: record.get(col, "") for col in columns}
        filtered_results.append(filtered)

    # Convert to DataFrame and write CSV
    df = pd.DataFrame(filtered_results, columns=columns)
    df.to_csv(file_path, index=False)
    logger.info(f"Saved {len(df)} coded rows to {file_path}")