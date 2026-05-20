# src/main.py

import yaml, logging
from datetime import datetime
from .data_loader import load_csv
from .chunker import get_chunker
from .coder import OpenCoder
from .writer import save_coded
from .utils import setup_logging, merge_config_with_env
from .filter import KeywordFilter

logger = logging.getLogger(__name__)


def main():
    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Merge environment variables (OLLAMA_MODEL, OLLAMA_HOST, etc.)
    cfg = merge_config_with_env(cfg)

    # Add timestamp to output filename
    output_path = cfg["output"]["file_path"]
    stem, ext = os.path.splitext(output_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg["output"]["file_path"] = f"{stem}_{timestamp}{ext}"

    # ------------------------------------------------------------------
    # 2. Setup logging (console + optional file)
    # ------------------------------------------------------------------
    setup_logging(cfg["logging"])

    # ------------------------------------------------------------------
    # 3. Initialise the keyword filter (if enabled)
    # ------------------------------------------------------------------
    filter_cfg = cfg.get("filter", {})
    if filter_cfg.get("enabled", True):
        keyword_filter = KeywordFilter.from_config(filter_cfg)
        logger.info("Keyword filter is ACTIVE")
    else:
        keyword_filter = None
        logger.info("Keyword filter is DISABLED – all chunks will be sent to LLM")

    # ------------------------------------------------------------------
    # 4. Load raw CSV data
    # ------------------------------------------------------------------
    rows = load_csv(cfg["input"])
    if not rows:
        logger.warning("No rows to process. Exiting.")
        return

    # ------------------------------------------------------------------
    # 5. Instantiate chunker and coder
    # ------------------------------------------------------------------
    chunker = get_chunker(cfg["chunking"])
    coder = OpenCoder(cfg["coding"])

    # ------------------------------------------------------------------
    # 6. Main pipeline: row → chunk(s) → LLM coding → results
    # ------------------------------------------------------------------
    coded_results = []
    total_rows = len(rows)
    total_chunks = 0
    skipped_chunks = 0

    for i, row in enumerate(rows, start=1):
        row_id = row["id"]
        text = row["text"]

        remaining = total_rows - i
        logger.info(f"Processing row {i}/{total_rows} (ID: {row_id}) – {remaining} rows left")

        chunks = chunker.chunk(text)
        if not chunks:
            logger.debug(f"No chunks generated for row {row_id}")
            continue

        for chunk in chunks:
            if keyword_filter and not keyword_filter.is_relevant(chunk):
                logger.debug(f"Skipping non‑relevant chunk: {chunk[:80]}...")
                coded_results.append(
                    {
                        "id": row_id,
                        "original_text": text,
                        "chunk": chunk,
                        "codes": "none",
                    }
                )
                skipped_chunks += 1
                continue
            
            codes = coder.code(chunk)
            if not codes:
                codes = "none"
            coded_results.append(
                {
                    "id": row_id,
                    "original_text": text,
                    "chunk": chunk,
                    "codes": codes,
                }
            )
            total_chunks += 1

            # Progress logging every 10 chunks
            if total_chunks % 10 == 0:
                logger.info(f"Processed {total_chunks} chunks so far...")

    logger.info(
        f"Finished. Rows: {total_rows}, "
        f"Chunks sent to LLM: {total_chunks}, "
        f"Chunks auto‑marked as 'none' (filter): {skipped_chunks}"
    )

    # ------------------------------------------------------------------
    # 7. Write coded output to CSV
    # ------------------------------------------------------------------
    save_coded(coded_results, cfg["output"])
    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()