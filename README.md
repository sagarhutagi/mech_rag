<<<<<<< HEAD
# mech_rag
=======
# 📐 Engineering Mechanics Statics RAG
### Built for: Meriam, Kraige & Bolton — 8th Edition

A fully local, offline RAG (Retrieval-Augmented Generation) system that lets you
query the textbook and solutions manual using natural language.

---

## 🗂️ Project Structure

```
statics_rag/
├── requirements.txt       # Python dependencies
├── 1_extract.py           # Step 1: Extract text + images from PDFs
├── 2_embed.py             # Step 2: Embed chunks into ChromaDB
├── 3_query.py             # Step 3: Query with local LLM
├── extracted/             # Auto-created: chunks JSON + images
└── chroma_db/             # Auto-created: vector database
```

---

## ⚙️ Setup

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Ollama (local LLM runner)
```bash
# Mac/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download from https://ollama.com
```

### 3. Pull the required models
```bash
# For math reasoning (recommended for statics)
ollama pull qwen2.5-math

# For vision OCR of handwritten solutions (pick one)
ollama pull qwen2-vl       # Best quality, needs ~8GB VRAM
ollama pull llava           # Lighter alternative

# Fallback text LLM
ollama pull llama3.1
```

---

## 🚀 Running the Pipeline

### Step 1 — Extract PDFs
Place your PDFs in the project folder and update paths in `1_extract.py`:
```python
TEXTBOOK_PDF  = "textbook.pdf"
SOLUTIONS_PDF = "solutions.pdf"
```

Then run:
```bash
python 1_extract.py
```
This creates `extracted/all_chunks.json` and saves solution page images.

**Tip:** First run with `use_vision_ocr=False` to test extraction speed without OCR.
Then flip it to `True` for the full handwriting recognition pass.

---

### Step 2 — Embed into ChromaDB
```bash
python 2_embed.py
```
This creates the `chroma_db/` folder with all embedded vectors.
Only needs to run **once** (or when you add new content).

---

### Step 3 — Query!
```bash
# Interactive mode
python 3_query.py

# Query a specific problem
python 3_query.py --problem "2/27"

# Ask a concept question
python 3_query.py --question "when do I use law of cosines vs law of sines for force triangles"
```

---

## 🧠 Architecture

```
PDF (Textbook)              PDF (Solutions - Handwritten)
      │                               │
  PyMuPDF                     qwen2-vl / llava (Ollama)
  Text Extraction             Vision OCR → clean text
      │                               │
      └──────────┬────────────────────┘
                 │
         Chunk by Problem Number
         (metadata: problem_id, chapter, source, page)
                 │
         Embed: all-MiniLM-L6-v2
         (sentence-transformers, local)
                 │
         ChromaDB (persistent, local)
                 │
         Query → Top-5 chunks retrieved
                 │
         qwen2.5-math (Ollama)
         Generates step-by-step explanation
```

---

## 🔑 Key Design Decisions

| Decision | Reason |
|---|---|
| ChromaDB | Zero setup, runs in-process, no server needed |
| all-MiniLM-L6-v2 | Fast, small, great for technical/math text |
| qwen2.5-math | Best local model for engineering math reasoning |
| qwen2-vl for OCR | Handles handwriting + equations better than Tesseract |
| Problem ID as metadata | Enables exact lookup ("show me 2/27") + semantic search |
| Images stored alongside | Fallback for when OCR fails or diagram context needed |

---

## 📈 Extending the System

**Add more textbooks:**
```python
# Just run extract + embed on a new PDF, same collection
TEXTBOOK_PDF = "dynamics.pdf"
```

**Add a web UI (Streamlit):**
```bash
pip install streamlit
# Build a simple chat interface on top of query.py
```

**Improve OCR accuracy:**
- Use Mathpix API instead of local vision model for better equation recognition
- Set `MATHPIX_APP_ID` and `MATHPIX_APP_KEY` in `.env`

---

## 🛠️ Troubleshooting

| Issue | Fix |
|---|---|
| Ollama model not found | Run `ollama pull <model_name>` |
| OCR is slow | Reduce DPI in `pdf_page_to_image(dpi=150)` or use `use_vision_ocr=False` |
| Problem IDs not detected | Check regex in `detect_problem_number()` and adjust for your PDF format |
| ChromaDB empty | Make sure `2_embed.py` completed without errors |
| Bad embedding quality | Try `all-mpnet-base-v2` instead of MiniLM for better accuracy |
>>>>>>> 7fab37a (all files)
