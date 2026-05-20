# src/main.py

import yaml, os, logging
from datetime import datetime
from src.data_loader import load_csv
from src.chunker import get_chunker
from src.coder import OpenCoder
from src.writer import save_coded
from src.utils import setup_logging, merge_config_with_env
from src.filter import KeywordFilter
from src.checkpoint import CheckpointManager


logger = logging.getLogger(__name__)

def run_profile(profile_name, profile_cfg, global_cfg, rows, chunker):
    # Load prompt
    prompt_path = profile_cfg.get("prompt_file")
    if not prompt_path or not os.path.exists(prompt_path):
        logger.error(f"Prompt file missing for profile {profile_name}: {prompt_path}")
        return
    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    coder_config = {**global_cfg["coding"], "prompt_template": prompt_template}
    coder = OpenCoder(coder_config)

    # Keyword filter setup
    filter_enabled = profile_cfg.get("filter_enabled", True)
    if filter_enabled:
        kw_file = profile_cfg.get("keywords_file")
        keywords = []
        if kw_file and os.path.exists(kw_file):
            with open(kw_file, "r") as f:
                keywords = [line.strip() for line in f if line.strip()]
        keyword_filter = KeywordFilter(keywords=keywords) if keywords else None
    else:
        keyword_filter = None

    # Output path
    output_dir = global_cfg.get("output_dir", "data/output")
    suffix = profile_cfg.get("output_suffix", profile_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{suffix}_{timestamp}.csv")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Profile '{profile_name}' will output to {output_path}")

    # Checkpoint setup
    checkpoint_cfg = global_cfg.get("checkpoint", {})
    checkpoint_dir = checkpoint_cfg.get("directory", "data/checkpoints")
    checkpoint_interval = checkpoint_cfg.get("interval", 10)
    ckpt = CheckpointManager(profile_name, checkpoint_dir)

    # Try to load existing checkpoint
    coded_results, last_row, last_chunk = ckpt.load()
    if coded_results:
        logger.info(f"Resuming from checkpoint: row {last_row}, chunk {last_chunk} ({len(coded_results)} results loaded)")
    else:
        last_row = 0
        last_chunk = -1

    total_rows = len(rows)
    total_chunks_sent_to_llm = 0
    skipped_chunks = 0
    chunk_counter_since_last_ckpt = 0

    for i in range(last_row, total_rows):
        row = rows[i]
        row_id = row["id"]
        text = row["text"]
        remaining = total_rows - i - 1
        logger.info(f"[{profile_name}] Row {i+1}/{total_rows} (ID: {row_id}) – {remaining} left")

        chunks = chunker.chunk(text)
        if not chunks:
            continue

        start_chunk = last_chunk + 1 if i == last_row else 0
        last_chunk = -1

        for j in range(start_chunk, len(chunks)):
            chunk = chunks[j]

            if keyword_filter and not keyword_filter.is_relevant(chunk):
                coded_results.append({
                    "id": row_id,
                    # "original_text": text,
                    "chunk": chunk,
                    "codes": "none"
                })
                skipped_chunks += 1
            else:
                codes = coder.code(chunk)
                coded_results.append({
                    "id": row_id,
                    # "original_text": text,
                    "chunk": chunk,
                    "codes": codes
                })
                total_chunks_sent_to_llm += 1

            chunk_counter_since_last_ckpt += 1

            # Save checkpoint at interval
            if checkpoint_interval > 0 and chunk_counter_since_last_ckpt >= checkpoint_interval:
                ckpt.save(coded_results, i, j)
                chunk_counter_since_last_ckpt = 0

        # Progress log every 10 chunks
        if (total_chunks_sent_to_llm + skipped_chunks) % 10 == 0:
            logger.info(f"  [{profile_name}] Total chunks handled: {total_chunks_sent_to_llm + skipped_chunks} (LLM: {total_chunks_sent_to_llm}, Skipped: {skipped_chunks})")

    # Final checkpoint save after all rows
    if checkpoint_interval > 0:
        # Save with the last row and last chunk indices
        # The last chunk processed is in row total_rows-1, index len(chunks)-1
        ckpt.save(coded_results, total_rows - 1, len(chunks) - 1 if chunks else 0)

    logger.info(f"[{profile_name}] Done. Sent to LLM: {total_chunks_sent_to_llm}, Skipped: {skipped_chunks}")

    # Save final output
    save_coded(coded_results, {"file_path": output_path})

    # Clear checkpoint files (successful run)
    ckpt.clear()

def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = merge_config_with_env(cfg)
    setup_logging(cfg["logging"])

    rows = load_csv(cfg["input"])
    if not rows:
        logger.warning("No rows. Exiting.")
        return

    chunker = get_chunker(cfg["chunking"])

    active_profiles = cfg.get("profiles", {}).get("active", [])
    if not active_profiles:
        logger.warning("No active profiles configured. Exiting.")
        return

    for profile_name in active_profiles:
        profile_cfg = cfg["profiles"].get(profile_name, {})
        if not profile_cfg:
            logger.warning(f"No config found for profile '{profile_name}'. Skipping.")
            continue
        run_profile(profile_name, profile_cfg, cfg, rows, chunker)

if __name__ == "__main__":
    main()