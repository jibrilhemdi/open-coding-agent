# app.py
import streamlit as st
import pandas as pd
import yaml
import logging
import os
from datetime import datetime
from pathlib import Path
import shutil

# Import your project modules (absolute imports work from project root)
from src.data_loader import load_csv
from src.chunker import get_chunker
from src.coder import OpenCoder
from src.filter import KeywordFilter
from src.writer import save_coded
from src.utils import setup_logging, merge_config_with_env

# ----------------------------------------------------------------------
# Profile management functions
# ----------------------------------------------------------------------

PROFILES_DIR = "profiles"

def load_profiles_list():
    """Return list of profile names (subfolders) inside profiles_dir."""
    p = Path(PROFILES_DIR)
    if not p.exists():
        return []
    return sorted([d.name for d in p.iterdir() if d.is_dir()])

def load_prompt_file(profile):
    """Return content of prompt.txt for a profile."""
    prompt_path = Path(PROFILES_DIR) / profile / "prompt.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""

def load_keywords_file(profile):
    """Return list of keywords from keywords.txt."""
    kw_path = Path(PROFILES_DIR) / profile / "keywords.txt"
    if kw_path.exists():
        return [line.strip() for line in kw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return []

def save_profile_files(profile, prompt_text, keywords_list):
    """Write prompt.txt and keywords.txt for a profile."""
    folder = Path(PROFILES_DIR) / profile
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "prompt.txt").write_text(prompt_text, encoding="utf-8")
    (folder / "keywords.txt").write_text("\n".join(keywords_list), encoding="utf-8")

def create_new_profile(name, prompt_text, keywords_text):
    """Create a new profile folder with the given name, prompt, and keywords (one per line)."""
    keywords = [kw.strip() for kw in keywords_text.split("\n") if kw.strip()]
    save_profile_files(name, prompt_text, keywords)

def delete_profile(name):
    """Delete a profile folder completely."""
    folder = Path(PROFILES_DIR) / name
    if folder.exists() and folder.is_dir():
        shutil.rmtree(folder)
        return True
    return False

# ----------------------------------------------------------------------
# Initialize session state for profile selection
# ----------------------------------------------------------------------
if "selected_profiles" not in st.session_state:
    st.session_state.selected_profiles = []

# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
st.set_page_config(page_title="Open‑Coding Agent", layout="wide")
st.title("🔍 Open‑Coding Agent (Ollama)")
st.markdown("""
**Open coding** is a qualitative method where raw text is segmented and tagged with meaningful labels.  
This tool lets you code large CSV files using a local LLM (Ollama) with custom profiles.  
For documentation and source code, visit the [GitHub repository](https://github.com/jibrilhemdi/open-coding-agent).
""")

# Load default config.yaml (if exists) for initial values
defaults = {}
if os.path.exists("config.yaml"):
    with open("config.yaml", "r") as f:
        defaults = yaml.safe_load(f) or {}

# Sidebar – settings & profile management
with st.sidebar:
    st.header("⚙️ Settings")

    # Model & connection
    model_name = st.text_input(
        "Ollama Model",
        value=defaults.get("coding", {}).get("model", "llama3.2")
    )
    temperature = st.slider(
        "Temperature",
        0.0, 1.0,
        value=defaults.get("coding", {}).get("temperature", 0.0),
        step=0.1
    )
    max_retries = st.number_input(
        "Max retries on failure",
        1, 10,
        value=defaults.get("coding", {}).get("max_retries", 3)
    )

    # Chunking
    chunk_strategy = st.selectbox(
        "Chunking strategy",
        ["sentence", "paragraph", "fixed_tokens"],
        index=0 if defaults.get("chunking", {}).get("strategy", "sentence") == "sentence" else 1
    )
    if chunk_strategy == "fixed_tokens":
        max_tokens = st.number_input(
            "Max tokens per chunk",
            64, 4096,
            value=defaults.get("chunking", {}).get("max_tokens", 512)
        )

    # Checkpoint
    st.subheader("Checkpoint")
    use_checkpoint = st.checkbox("Enable checkpointing", value=True)
    checkpoint_interval = st.number_input(
        "Save every N chunks",
        1, 100,
        value=defaults.get("checkpoint", {}).get("interval", 10)
    )

    st.divider()

    # ------------------------------------------------------------------
    # Profile selection
    # ------------------------------------------------------------------
    st.header("🎯 Profiles")

    # Get available profiles from disk
    available_profiles = load_profiles_list()

    # Display current profiles or a message
    if not available_profiles:
        st.info("No profiles found. Create one below.")

    # Profile multiselect (session state controlled)
    selected_profiles = st.multiselect(
        "Choose profiles to run",
        options=available_profiles,
        default=st.session_state.selected_profiles,
        key="profile_multiselect"
    )
    # Update session state if user changes selection manually
    st.session_state.selected_profiles = selected_profiles

    # ------------------------------------------------------------------
    # Create new profile
    # ------------------------------------------------------------------
    with st.expander("➕ Create New Profile"):
        with st.form("new_profile_form"):
            new_name = st.text_input("Profile name (folder name)")
            new_prompt = st.text_area("Prompt template", 
                "You are doing open coding...\n\nText to analyse: {text}",
                height=150)
            new_keywords = st.text_area("Keywords (one per line)", 
                placeholder="keyword1\nkeyword2")
            submitted = st.form_submit_button("Create Profile")
            if submitted and new_name.strip():
                # Validate name – no spaces/special chars
                safe_name = new_name.strip().replace(" ", "_").lower()
                if safe_name in available_profiles:
                    st.error(f"Profile '{safe_name}' already exists!")
                else:
                    create_new_profile(safe_name, new_prompt, new_keywords)
                    st.success(f"Profile '{safe_name}' created!")
                    st.rerun()  # refresh to show new profile

# ----------------------------------------------------------------------
# Main area – data upload & profile preview
# ----------------------------------------------------------------------
st.subheader("📄 Upload CSV")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
text_column = st.text_input(
    "Name of the text column",
    value=defaults.get("input", {}).get("text_column", "text")
)

# Show prompts / keywords for selected profiles (with ability to edit and save)
if selected_profiles:
    st.subheader("📝 Profile previews (editable & saveable)")
    tabs = st.tabs(selected_profiles)
    for i, profile in enumerate(selected_profiles):
        with tabs[i]:
            prompt = load_prompt_file(profile)
            edited_prompt = st.text_area(
                f"Prompt for '{profile}'",
                prompt,
                height=200,
                key=f"prompt_{profile}"
            )
            keywords = load_keywords_file(profile)
            keywords_text = "\n".join(keywords)
            edited_keywords = st.text_area(
                f"Keywords for '{profile}' (one per line)",
                keywords_text,
                height=100,
                key=f"keywords_{profile}"
            )
            use_filter = st.checkbox(
                f"Use keyword filter for '{profile}'",
                value=True,
                key=f"filter_{profile}"
            )
            # Save button for this profile
            if st.button(f"💾 Save changes to '{profile}' files", key=f"save_{profile}"):
                kw_list = [kw.strip() for kw in edited_keywords.split("\n") if kw.strip()]
                save_profile_files(profile, edited_prompt, kw_list)
                st.success(f"Profile '{profile}' updated on disk.")

            # --- Delete profile section ---
            if f"delete_confirm_{profile}" not in st.session_state:
                st.session_state[f"delete_confirm_{profile}"] = False

            # Button to initiate deletion
            if st.button(f"🗑️ Delete '{profile}' profile", key=f"del_{profile}"):
                st.session_state[f"delete_confirm_{profile}"] = True

            # Show confirmation warning if delete was triggered
            if st.session_state[f"delete_confirm_{profile}"]:
                st.warning(f"Are you sure you want to delete the profile **{profile}**? This cannot be undone.")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("✅ Yes, delete", key=f"confirm_del_{profile}"):
                        if delete_profile(profile):
                            # Remove profile from current selection
                            if profile in st.session_state.selected_profiles:
                                st.session_state.selected_profiles.remove(profile)
                            st.success(f"Profile '{profile}' deleted.")
                            st.session_state[f"delete_confirm_{profile}"] = False
                            st.rerun()
                with col_no:
                    if st.button("❌ Cancel", key=f"cancel_del_{profile}"):
                        st.session_state[f"delete_confirm_{profile}"] = False
                        st.rerun()

# Run button
run_btn = st.button(
    "🚀 Run Coding",
    type="primary",
    disabled=not uploaded_file or not selected_profiles
)

# ----------------------------------------------------------------------
# Execution logic
# ----------------------------------------------------------------------
if run_btn:
    with st.status("Coding in progress...", expanded=True) as status:
        try:
            df = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
            if text_column not in df.columns:
                st.error(f"Column '{text_column}' not found in CSV.")
                st.stop()

            id_col = "id" if "id" in df.columns else None
            rows = []
            for idx, row in df.iterrows():
                rid = row[id_col] if id_col else str(idx)
                rows.append({"id": rid, "text": row[text_column]})
            st.write(f"Loaded {len(rows)} rows.")

            chunker_config = {"strategy": chunk_strategy}
            if chunk_strategy == "fixed_tokens":
                chunker_config["max_tokens"] = max_tokens
            chunker = get_chunker(chunker_config)

            for profile in selected_profiles:
                st.write(f"### Coding profile: **{profile}**")
                progress_bar = st.progress(0, text=f"{profile}: 0%")
                profile_output = []

                # Use edited values from session_state
                prompt_template = st.session_state[f"prompt_{profile}"]
                kw_text = st.session_state[f"keywords_{profile}"]
                keywords = [kw.strip() for kw in kw_text.split("\n") if kw.strip()]
                use_filter = st.session_state[f"filter_{profile}"]

                coder_config = {
                    "model": model_name,
                    "temperature": temperature,
                    "max_retries": max_retries,
                    "prompt_template": prompt_template
                }
                coder = OpenCoder(coder_config)

                filter_enabled = use_filter and bool(keywords)
                keyword_filter = KeywordFilter(keywords=keywords) if filter_enabled else None

                total_rows = len(rows)
                for i, row in enumerate(rows):
                    row_id = row["id"]
                    text = row["text"]
                    chunks = chunker.chunk(text)

                    progress_percent = int((i + 1) / total_rows * 100)
                    progress_bar.progress(progress_percent, text=f"{profile}: row {i+1}/{total_rows}")

                    for chunk in chunks:
                        if keyword_filter and not keyword_filter.is_relevant(chunk):
                            code = "none"
                        else:
                            code = coder.code(chunk)
                        profile_output.append({
                            "id": row_id,
                            "original_text": text,
                            "chunk": chunk,
                            "codes": code
                        })

                st.success(f"Profile '{profile}' completed. Chunks: {len(profile_output)}")

                if profile_output:
                    df_out = pd.DataFrame(profile_output)
                    csv = df_out.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"⬇️ Download {profile} results CSV",
                        data=csv,
                        file_name=f"{profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    with st.expander("Preview first 10 rows"):
                        st.dataframe(df_out.head(10))

            status.update(label="Coding complete!", state="complete")

        except Exception as e:
            status.update(label="Error occurred", state="error")
            st.exception(e)