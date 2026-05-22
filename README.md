# Open‑Coding Agent (Local with Ollama)

A modular, extensible Python pipeline that uses a local LLM (via [Ollama](https://ollama.com)) to perform **qualitative open coding (semi‑inductive)** on raw text from a CSV file.

You can run the pipeline from the command line or through an interactive [Streamlit web interface](https://open-coding-agent.streamlit.app/).

Designed for researchers and analysts who need to:

- Segment large text datasets into chunks (sentences, paragraphs, or fixed‑size)
- Apply **multiple coding profiles** simultaneously (e.g. "role‑play themes" and "embarrassment cues")
- Use a configurable **keyword filter** to skip irrelevant chunks before calling the LLM (saving time)
- **Resume from the last checkpoint** if the pipeline is interrupted – no lost work
- Output timestamped CSV files with rich, context‑specific codes

---

## What is Open Coding?

Open coding is a foundational technique in qualitative research. It involves breaking raw text data into small segments (chunks) and assigning concise labels (codes) that capture the underlying meaning, topic, emotion, etc.  

This project automates that process using a **local LLM** (via [Ollama](https://ollama.com)).  
You define a coding focus (e.g., role‑play dynamics, embarrassment cues, user motivations) through a simple prompt and a keyword filter, and the pipeline returns a coded CSV ready for analysis.

---

## Features

- **Multi‑profile** – run one or several coding tasks in a single pass, each with its own prompt and keywords.
- **Local LLM** – all coding runs locally through Ollama; no external API calls.
- **Custom prompts** – place your coding instructions in separate text files; change them without touching Python.
- **Keyword pre‑filter** – speed up processing by skipping chunks that don’t contain domain‑specific keywords.
- **Checkpointing** – periodically saves progress; restart from exactly where you left off if something fails.
- **Timestamped output** – every run creates a new CSV file; never overwrite previous results.
- **Pluggable chunking** – sentence, paragraph, or fixed‑token chunking strategies (easily extendable).
- **Streamlit GUI** – upload data, manage profiles, edit prompts, and run the pipeline from your browser.

---

## Project Structure

```
open-coding-agent/
├── README.md
├── requirements.txt
├── config.yaml                 # main configuration file
├── app.py                      # Streamlit UI
├── prompts/                    # (optional) default prompt templates
├── profiles/                   # coding profiles (one folder per theme)
│   ├── roleplay/
│   │   ├── prompt.txt
│   │   └── keywords.txt
│   └── embarrassment/
│       ├── prompt.txt
│       └── keywords.txt
├── src/
│   ├── main.py                 # entry point – orchestrates the pipeline
│   ├── data_loader.py          # reads CSV and validates input
│   ├── chunker.py              # text segmentation strategies
│   ├── coder.py                # Ollama LLM interaction
│   ├── filter.py               # keyword‑based pre‑filter
│   ├── checkpoint.py           # resumeable progress
│   ├── writer.py               # saves coded results to CSV
│   └── utils.py                # logging, config merging
├── data/
│   ├── input/                  # place raw CSV files here
│   ├── output/                 # coded output appears here
│   └── checkpoints/            # automatic checkpoint files
└── tests/                      # unit and integration tests
```

---

## Prerequisites

- **Ollama** installed and running locally.  
  Download from [ollama.com](https://ollama.com) and pull a model (e.g. `gemma4:e2b`):
  ```bash
  ollama serve          # start the server (in a separate terminal)
  ollama pull gemma4:e2b
  ```
- **Python 3.9+** with a virtual environment recommended.

---

## Setup

1. **Clone or download** the project and navigate into it:
   ```bash
   cd open-coding-agent
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # macOS/Linux
   # .venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your input CSV** – place it in `data/input/`.  
   The CSV must contain at least one column with the raw text (e.g. a column named `review`).  
   An optional `id` column can be used to identify rows; otherwise row numbers are used.

---

## Streamlit Web Interface (Recommended)

For an **interactive, no‑code experience**, launch the GUI:

```bash
streamlit run app.py
```

Your browser will open `http://localhost:8501`. The app allows you to:

- 📄 **Upload a CSV** file directly.
- 🎯 **Select existing profiles** or **create new ones** from the sidebar.
- 📝 **Preview and edit prompts/keywords** in real time, and save changes to disk.
- 🗑️ **Delete** unwanted profiles with a single click.
- ⚙️ **Configure** the model, chunking, checkpointing, and filter settings.
- 🚀 **Run the coding** and watch live progress bars.
- ⬇️ **Download** separate CSV outputs for each profile.

All changes you make to prompts and keywords in the UI can be saved permanently, making it easy to iterate on your coding scheme without touching a text editor.

---

## Configuration

All behaviour is controlled through `config.yaml`. Here’s a minimal example with comments:

```yaml
input:
  file_path: "data/input/sample.csv"
  text_column: "review"          # name of the column containing raw text

chunking:
  strategy: "sentence"           # options: sentence, paragraph, fixed_tokens
  max_tokens: 512                # only used when strategy = fixed_tokens

coding:
  model: "gemma4:e2b"             # the Ollama model you have pulled
  temperature: 0.0
  max_retries: 3

output:
  # The base output path – a timestamp will be inserted automatically.
  file_path: "data/output/coded_output.csv"

logging:
  level: "INFO"
  file: "logs/pipeline.log"      # if omitted, logs only to console

checkpoint:
  directory: "data/checkpoints"
  interval: 10                   # save after every 10 chunks (set to 0 to disable)

profiles:
  active:                        # profiles to run (can be one or many)
    - roleplay
    - embarrassment

  output_dir: "data/output"      # overrides output.file_path directory for profiles
  # (the global output.file_path above is only used if no profiles are active)

  roleplay:
    prompt_file: "profiles/roleplay/prompt.txt"
    keywords_file: "profiles/roleplay/keywords.txt"
    filter_enabled: true
    output_suffix: "rp"

  embarrassment:
    prompt_file: "profiles/embarrassment/prompt.txt"
    keywords_file: "profiles/embarrassment/keywords.txt"
    filter_enabled: true
    output_suffix: "embarrass"
```

### Profile files

Each profile folder (e.g. `profiles/embarrassment/`) contains:

- **`prompt.txt`** – the instructions for the LLM, including code examples.  
  The placeholder `{text}` is replaced by the chunk at runtime.
- **`keywords.txt`** (optional) – one keyword per line. Chunks that don’t contain any keyword are skipped (coded as `none`) without sending them to the LLM.

You can add as many profiles as you like; just create a new folder and add an entry in `config.yaml`.

---

## Running the Pipeline (Optional)

From the project root:

```bash
python -m src.main
```

The pipeline will:

1. Load the CSV.
2. For each **active profile**, chunk every row and code the relevant chunks.
3. Save a timestamped CSV per profile in `data/output/` (e.g. `rp_20260123_154512.csv`).

Progress is logged to the console (and optionally to a file).  
If the `checkpoint.interval` is > 0, progress is saved periodically and you can interrupt/resume safely.

### Resume after interruption

Just re‑run the same command. The pipeline automatically detects a previous checkpoint for each profile and continues from the last saved chunk – no manual steps needed.

---

## Output Format

The output CSV has three columns:

| Column        | Description |
|---------------|-------------|
| `id`          | Row identifier (from the original CSV’s `id` column, or the row index) |
| `chunk`       | The chunk of text that was coded |
| `codes`       | Comma‑separated codes assigned by the LLM, or `none` if the chunk was filtered/off‑topic |


---

## Customizing for Your Own Coding Tasks

1. **Create a new profile**:
   ```bash
   mkdir -p profiles/my_theme
   ```
2. Write your **prompt** in `profiles/my_theme/prompt.txt`.  
   See the existing profiles for examples on how to instruct the model to produce specific codes.
3. (Optional) Create a **keyword filter** file `profiles/my_theme/keywords.txt` to pre‑filter chunks.
4. Add the profile to `config.yaml` under `profiles.active` and define its settings.

---

## Dependencies

- `pandas` – data handling
- `pyyaml` – configuration parsing
- `python-dotenv` – environment variable loading (optional)
- `nltk` – sentence tokenization (optional; `punkt` downloaded automatically)
- `ollama` – Python client for Ollama

Install all with `pip install -r requirements.txt`.
