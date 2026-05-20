# src/checkpoint.py
"""
Simple checkpoint manager that saves/loads partial results
and the current position (row index, chunk index) for a given profile.
"""

import json
import os
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, profile_name: str, directory: str):
        self.profile = profile_name
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

        self.state_path = self.dir / f".state_{profile_name}.json"
        self.partial_path = self.dir / f".partial_{profile_name}.csv"

    def save(self, coded_results: List[Dict[str, Any]], row_idx: int, chunk_idx: int) -> None:
        """
        Save the current state.

        Args:
            coded_results: all chunks coded so far (will be written to the partial CSV)
            row_idx: index of the row (in the original rows list) that the **last processed chunk** belongs to
            chunk_idx: index of that chunk within the row (0‑based)
        """
        # 1. Save the coded results as CSV
        df = pd.DataFrame(coded_results)
        df.to_csv(self.partial_path, index=False)

        # 2. Save the state (last completed position)
        state = {"last_row_idx": row_idx, "last_chunk_idx": chunk_idx}
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f)
        logger.debug(f"Checkpoint saved: row {row_idx}, chunk {chunk_idx} ({len(coded_results)} results)")

    def load(self) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Load a previous checkpoint if it exists.

        Returns:
            (coded_results, next_row_start, next_chunk_start)
            If no checkpoint exists, returns ([], 0, 0).
        """
        if not self.state_path.exists() or not self.partial_path.exists():
            return [], 0, 0

        # Load state
        with open(self.state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        last_row = state["last_row_idx"]
        last_chunk = state["last_chunk_idx"]

        # Load partial results
        df = pd.read_csv(self.partial_path, dtype=str, keep_default_na=False)
        coded_results = df.to_dict(orient="records")

        # Compute where to resume: right after the last processed chunk
        # We'll need to know how many chunks that row has, so we can't just blindly
        # set next_chunk = last_chunk+1; the caller will handle that.
        # Here we return the last completed position – the caller will advance.
        return coded_results, last_row, last_chunk

    def clear(self) -> None:
        """Remove checkpoint files after successful completion."""
        for path in (self.state_path, self.partial_path):
            if path.exists():
                path.unlink()
                logger.debug(f"Removed {path}")