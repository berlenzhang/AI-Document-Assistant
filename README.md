# AI Document Assistant

A RAG (Retrieval-Augmented Generation) document assistant. Upload PDF, DOCX, or TXT files and ask questions about them — answers come with citations back to the source document and page.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

No API key is required by default — the app runs fully locally using a local embedding model and a local flan-t5 model for answers.

## Running the app

**Backend** (from `backend/`):
```bash
source ../venv/bin/activate
uvicorn main:app --reload --port 8000
```

**Frontend** (from `frontend/`):
```bash
python3 -m http.server 3000
```

Then open `http://localhost:3000` in your browser. Interactive API docs are available at `http://localhost:8000/docs`.

## Configuration

Copy `.env.example` to `.env` and edit as needed. All settings are optional — the app works out of the box.

Notable settings:
- `LLM_PROVIDER` — `flan_t5` (default, fully local) or `claude` (uses the Anthropic API — requires `ANTHROPIC_API_KEY`).
- Changes to `.env` require a backend restart to take effect.

## Evaluation

The `eval/` directory has a small script-based harness for checking retrieval quality and answer faithfulness:

```bash
python3 eval/retrieval_eval.py       # free, local, no API key needed
python3 eval/faithfulness_eval.py    # requires ANTHROPIC_API_KEY, makes real API calls
python3 eval/run_all.py              # runs both, merges into eval/report.md
```

## Project structure

```
backend/    FastAPI app — upload, query, and document management endpoints
frontend/   Static HTML/JS UI
eval/       Evaluation scripts and hand-labeled test set
data/       Uploaded documents and the vector store (created automatically)
```

See `CLAUDE.md` for architecture details and design notes.
