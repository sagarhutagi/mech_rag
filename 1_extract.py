"""
STEP 1: extract.py

Three sources:
  1. Textbook   — theory + examples (typed PDF, one big file)
  2. Question Bank (QB) — clean questions by topic (typed PDFs, one per topic)
  3. Solutions  — handwritten, one PDF per problem

Metadata is derived entirely from folder/file names — no guessing needed.

After extraction, chunks are cross-linked:
  QB question  ←→  Solution  (matched by chapter + problem number)
  Textbook page ←→ QB question (matched by chapter)
"""

import fitz  # PyMuPDF
import os
import json
import re
from pathlib import Path
from PIL import Image
import io
import base64
from tqdm import tqdm
import ollama

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SOURCE_DIR     = "source_files"
TEXTBOOK_PDF   = os.path.join(SOURCE_DIR, "Engineering Mechanics Statics 8 edition Text book.pdf")
QB_DIR         = os.path.join(SOURCE_DIR, "Question Bank")
SOLUTIONS_DIR  = os.path.join(SOURCE_DIR, "Solutions for QB")
OUTPUT_DIR     = "extracted"
IMAGES_DIR     = os.path.join(OUTPUT_DIR, "solution_images")

VISION_MODEL   = "glm-ocr:latest"    # or "llava" — for handwritten OCR
USE_VISION_OCR = True          # ← set True for full OCR pass (slow but accurate)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


# ─── METADATA PARSERS ─────────────────────────────────────────────────────────

# QB filename examples:
#   "1 U1 Force Numericals.pdf"          → unit=1, topic=Force
#   "5 U2 Equilibrium Numericals.pdf"    → unit=2, topic=Equilibrium
#   "10 U4 Moment of Inertia Area Numericals.pdf" → unit=4, topic=Moment of Inertia Area
QB_PATTERN = re.compile(r'(?:\d+\s+)?U(\d+)\s+(.+?)\s+Numericals', re.IGNORECASE)

# Solution folder examples:
#   "1 U1 Force Solution"   → unit=1, topic=Force
SOL_FOLDER_PATTERN = re.compile(r'(?:\d+\s+)?U(\d+)\s+(.+?)\s+Solution', re.IGNORECASE)

# Solution filename: "2 - 27.pdf" → ch=2, prob=27 | "A - 16.pdf" → ch=A, prob=16
FILENAME_PATTERN = re.compile(r'([A-Za-z0-9]+)\s*-\s*(\d+)')


def parse_qb_filename(name: str) -> dict:
    m = QB_PATTERN.search(name)
    if m:
        return {"unit": int(m.group(1)), "topic": m.group(2).strip()}
    return {"unit": 0, "topic": "Unknown"}


def parse_solution_folder(name: str) -> dict:
    m = SOL_FOLDER_PATTERN.search(name)
    if m:
        return {"unit": int(m.group(1)), "topic": m.group(2).strip()}
    return {"unit": 0, "topic": "Unknown"}


def parse_solution_filename(filename: str) -> dict:
    stem = Path(filename).stem
    m = FILENAME_PATTERN.match(stem)
    if m:
        ch_raw = m.group(1).strip()
        prob   = int(m.group(2).strip())
        ch_num = "A" if ch_raw.upper() == "A" else ch_raw
        return {
            "ch_num":     ch_num,
            "chapter":    "Appendix" if ch_num == "A" else f"Chapter {ch_num}",
            "problem":    prob,
            "problem_id": f"ch{ch_num}_prob{prob}",
        }
    return {"ch_num": "?", "chapter": "Unknown", "problem": 0, "problem_id": "unknown"}


# ─── VISION OCR (for handwritten solutions) ──────────────────────────────────

def pdf_page_to_image(page, dpi=200) -> Image.Image:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ocr_solution_page(img: Image.Image, problem_id: str, topic: str) -> str:
    b64 = image_to_base64(img)
    prompt = f"""You are extracting handwritten engineering mechanics solutions.

Problem ID: {problem_id}
Topic: {topic}

Extract exactly what is visible on the page.

Rules:
- Preserve all equations and symbols
- Convert equations into readable LaTeX-style text
- Preserve units exactly
- Preserve numbered steps if present
- Describe diagrams as:
  [FBD: ...]
- Mark final answer lines as:
  ANSWER: ...
- If something is unclear, write [unclear]
- Do not solve, infer, or add missing content
- Do not explain anything

Output only the extracted content."""

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": prompt, "images": [b64]}]
    )
    return response["message"]["content"].strip()


# ─── SOURCE 1: QUESTION BANK ──────────────────────────────────────────────────

def extract_question_bank() -> list[dict]:
    """
    Extract questions from QB PDFs.
    Each QB PDF is typed → native text extraction works well.
    We attempt to split by individual problem numbers within the PDF.
    """
    qb_path = Path(QB_DIR)
    if not qb_path.exists():
        print(f"  QB dir not found: {QB_DIR}")
        return []

    qb_files = sorted(qb_path.glob("*.pdf"))
    print(f"\n  Found {len(qb_files)} QB files")

    all_chunks = []

    for pdf_file in tqdm(qb_files, desc="Question Bank"):
        meta   = parse_qb_filename(pdf_file.name)
        doc    = fitz.open(str(pdf_file))
        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + "\n"
        doc.close()

        # Try to split by problem numbers (e.g. "2/27", "2-27", "Problem 2.27")
        # QB problems are typically listed sequentially
        # Split on patterns like  "2/1", "2/27", "3-45" at start of line
        segments = re.split(r'(?m)(?=^\s*\d+[/\-]\d+\b)', full_text)

        parsed_any = False
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            m = re.match(r'(\d+)[/\-](\d+)', segment)
            if m:
                ch_num     = m.group(1)
                prob_num   = int(m.group(2))
                problem_id = f"ch{ch_num}_prob{prob_num}"
                all_chunks.append({
                    "problem_id":  problem_id,
                    "ch_num":      ch_num,
                    "chapter":     f"Chapter {ch_num}",
                    "problem":     prob_num,
                    "unit":        meta["unit"],
                    "topic":       meta["topic"],
                    "text":        segment,
                    "source":      "question_bank",
                    "qb_file":     pdf_file.name,
                })
                parsed_any = True

        # Fallback: store the entire QB file as one chunk if no splits found
        if not parsed_any and full_text.strip():
            all_chunks.append({
                "problem_id":  f"qb_{meta['topic'].replace(' ','_').lower()}",
                "ch_num":      "?",
                "chapter":     "Unknown",
                "problem":     None,
                "unit":        meta["unit"],
                "topic":       meta["topic"],
                "text":        full_text.strip(),
                "source":      "question_bank",
                "qb_file":     pdf_file.name,
            })

    print(f"  Extracted {len(all_chunks)} QB question chunks")
    return all_chunks


# ─── SOURCE 2: SOLUTIONS ──────────────────────────────────────────────────────

def extract_all_solutions() -> list[dict]:
    """
    One PDF = one problem. Filename encodes chapter + problem number.
    Folder encodes unit + topic.
    """
    solutions_path = Path(SOLUTIONS_DIR)
    if not solutions_path.exists():
        print(f"  Solutions dir not found: {SOLUTIONS_DIR}")
        return []

    all_pdfs = []
    for unit_dir in sorted(solutions_path.iterdir()):
        if not unit_dir.is_dir():
            continue
        for topic_dir in sorted(unit_dir.iterdir()):
            if not topic_dir.is_dir():
                continue
            for pdf_file in sorted(topic_dir.glob("*.pdf")):
                all_pdfs.append((unit_dir, topic_dir, pdf_file))

    print(f"\n  Found {len(all_pdfs)} solution PDFs")
    all_chunks = []

    OCR_CACHE_DIR = os.path.join(OUTPUT_DIR, "ocr_cache")
    os.makedirs(OCR_CACHE_DIR, exist_ok=True)

    for unit_dir, topic_dir, pdf_file in tqdm(all_pdfs, desc="Solutions"):
        topic_meta = parse_solution_folder(topic_dir.name)
        file_meta  = parse_solution_filename(pdf_file.name)

        unit_m   = re.search(r'Unit\s+(\d+)', unit_dir.name, re.IGNORECASE)
        unit_num = int(unit_m.group(1)) if unit_m else topic_meta.get("unit", 0)

        safe_name = pdf_file.stem.replace(" ", "_")

        doc  = fitz.open(str(pdf_file))
        all_page_texts = []
        all_image_paths = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            img = pdf_page_to_image(page, dpi=200)

            page_img_path = os.path.join(IMAGES_DIR, f"sol_{safe_name}_p{page_idx+1}.png")
            img.save(page_img_path)
            all_image_paths.append(page_img_path)

            if USE_VISION_OCR:
                cache_file = os.path.join(OCR_CACHE_DIR, f"{safe_name}_p{page_idx+1}.txt")
                if os.path.exists(cache_file):
                    with open(cache_file, "r", encoding="utf-8") as f:
                        page_text = f.read()
                else:
                    page_text = ocr_solution_page(img, file_meta["problem_id"], topic_meta["topic"])
                    with open(cache_file, "w", encoding="utf-8") as f:
                        f.write(page_text)
            else:
                native = page.get_text("text").strip()
                page_text = native if native else "[Image — OCR not run]"
            
            all_page_texts.append(f"[PAGE {page_idx+1}]\n{page_text}")

        ocr_text = "\n\n".join(all_page_texts)

        all_chunks.append({
            "problem_id":   file_meta["problem_id"],
            "ch_num":       file_meta["ch_num"],
            "chapter":      file_meta["chapter"],
            "problem":      file_meta["problem"],
            "unit":         unit_num,
            "topic":        topic_meta["topic"],
            "unit_folder":  unit_dir.name,
            "topic_folder": topic_dir.name,
            "text":         ocr_text,
            "source":       "solution",
            "pdf_path":     str(pdf_file),
            "image_paths":  all_image_paths,
            "page_count":   len(doc),
        })
        
        doc.close()

    return all_chunks


# ─── SOURCE 3: TEXTBOOK ───────────────────────────────────────────────────────

def split_textbook_text(text, max_chars=900, overlap=150):
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""

    for p in paras:
        if len(current) + len(p) + 1 <= max_chars:
            current += ("\n" if current else "") + p
        else:
            if current:
                chunks.append(current)
            current = current[-overlap:] + "\n" + p if current else p

    if current:
        chunks.append(current)

    return chunks

def extract_textbook() -> list[dict]:
    """
    Extract textbook by page. Typed text → native extraction.
    Each chunk = one page with chapter metadata.
    """
    if not os.path.exists(TEXTBOOK_PDF):
        print(f"  Textbook not found: {TEXTBOOK_PDF}")
        return []

    print(f"\n  Extracting textbook...")
    doc = fitz.open(TEXTBOOK_PDF)
    chunks = []
    current_chapter = 1

    for page_num in tqdm(range(len(doc)), desc="Textbook pages"):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if not text:
            continue

        ch_match = re.search(r'CHAPTER\s+(\d+)', text, re.IGNORECASE)
        if ch_match:
            current_chapter = int(ch_match.group(1))

        # Save any embedded diagrams
        images_on_page = []
        for idx, img_info in enumerate(page.get_images(full=True)):
            base_image = doc.extract_image(img_info[0])
            img_save   = os.path.join(IMAGES_DIR, f"tb_ch{current_chapter}_p{page_num+1}_{idx}.png")
            with open(img_save, "wb") as f:
                f.write(base_image["image"])
            images_on_page.append(img_save)

        subchunks = split_textbook_text(text)

        for idx, sub in enumerate(subchunks):
            chunks.append({
                "problem_id": f"tb_ch{current_chapter}_p{page_num+1}_c{idx+1}",
                "ch_num": str(current_chapter),
                "chapter": f"Chapter {current_chapter}",
                "problem": None,
                "unit": None,
                "topic": None,
                "text": sub,
                "source": "textbook",
                "page": page_num + 1,
                "chunk_index": idx + 1,
                "image_paths": images_on_page,
            })

    print(f"  Extracted {len(chunks)} textbook page chunks")
    return chunks


# ─── CROSS-LINK: QB ←→ SOLUTION ───────────────────────────────────────────────

def cross_link(qb_chunks: list, sol_chunks: list) -> tuple[list, list]:
    """
    For each QB question chunk, find its matching solution chunk by problem_id.
    Adds 'has_solution': True/False and 'solution_image_path' to QB chunks.
    Adds 'has_question': True/False to solution chunks.
    """
    sol_index = {c["problem_id"]: c for c in sol_chunks}
    qb_index  = {c["problem_id"]: c for c in qb_chunks}

    linked = 0
    for qb in qb_chunks:
        pid = qb["problem_id"]
        if pid in sol_index:
            qb["has_solution"]       = True
            qb["solution_image_paths"] = sol_index[pid].get("image_paths", [])
            sol_index[pid]["has_question"] = True
            linked += 1
        else:
            qb["has_solution"] = False

    for sol in sol_chunks:
        if "has_question" not in sol:
            sol["has_question"] = False

    print(f"\n  Cross-linked {linked} QB questions ↔ solutions")
    unmatched_qb  = sum(1 for q in qb_chunks  if not q["has_solution"])
    unmatched_sol = sum(1 for s in sol_chunks  if not s["has_question"])
    if unmatched_qb:
        print(f"  {unmatched_qb} QB questions have no matching solution (may be extra problems)")
    if unmatched_sol:
        print(f"  {unmatched_sol} solutions have no matching QB question (may be extras)")

    return qb_chunks, sol_chunks


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("Engineering Mechanics Statics RAG — Extractor")
    print(f"Sources : Textbook + Question Bank ({QB_DIR}) + Solutions")
    print(f"OCR     : {'Vision (' + VISION_MODEL + ')' if USE_VISION_OCR else 'Native only — set USE_VISION_OCR=True for full run'}")
    print("=" * 65)

    # Extract all three sources
    qb_chunks  = extract_question_bank()
    sol_chunks = extract_all_solutions()
    tb_chunks  = extract_textbook()

    # Cross-link QB ↔ solutions
    qb_chunks, sol_chunks = cross_link(qb_chunks, sol_chunks)

    # Merge all
    all_chunks = qb_chunks + sol_chunks + tb_chunks

    # Save
    out_path = os.path.join(OUTPUT_DIR, "all_chunks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    from collections import Counter
    by_source = Counter(c["source"] for c in all_chunks)

    print(f"\n{'='*65}")
    print("EXTRACTION COMPLETE")
    print(f"  Question Bank chunks : {by_source['question_bank']}")
    print(f"  Solution chunks      : {by_source['solution']}")
    print(f"  Textbook chunks      : {by_source['textbook']}")
    print(f"  TOTAL                : {len(all_chunks)}")
    print(f"  Saved to             : {out_path}")

    print(f"\nBreakdown by topic (solutions):")
    topics = Counter(c["topic"] for c in sol_chunks)
    for topic, count in sorted(topics.items(), key=lambda x: -x[1]):
        print(f"  {topic:<40} {count} problems")

    print(f"\nNext step: python 2_embed.py")
