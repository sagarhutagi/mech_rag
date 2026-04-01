"""
STEP 3: query.py

Two core fixes:
  1. DB startup: get_or_create_collection instead of get_collection
  2. Math rendering: pattern-based detector — works even when LLM
     ignores $...$ instructions (which it often does)
"""

import re
import os
import sys
import argparse

# Suppress HuggingFace and transformers warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import base64

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.columns import Columns
from rich.prompt import Prompt
from rich.rule import Rule
from rich.live import Live
from rich import box
from rich.align import Align
from rich.padding import Padding

from pylatexenc.latex2text import LatexNodes2Text

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CHROMA_DIR   = "./chroma_db"
COLLECTION   = "statics_8th_edition"
EMBED_MODEL  = "all-MiniLM-L6-v2"
N_RESULTS    = 6
MAX_HISTORY  = 10

# Globals set during setup
LLM_MODEL    = "qwen/qwen3.6-plus-preview:free"
LLM_PROVIDER = "OpenRouter"

console = Console()
_latex  = LatexNodes2Text()

SOURCE_STYLE = {
    "solution":      ("bold green",   "✦ SOLUTION"),
    "question_bank": ("bold blue",    "? QUESTION"),
    "textbook":      ("bold magenta", "📖 TEXTBOOK"),
}

# ═══════════════════════════════════════════════════════════════════════════════
#  MATH RENDERING
#  Strategy: two-pass
#    Pass 1 — convert explicit $...$ LaTeX to Unicode
#    Pass 2 — detect equation-like patterns in plain text and highlight them
# ═══════════════════════════════════════════════════════════════════════════════

# Greek + symbol substitution table
SYMBOLS = [
    # Greek lowercase
    (r'\Sigma', 'Σ'), (r'\sigma', 'σ'), (r'\theta', 'θ'), (r'\Theta', 'Θ'),
    (r'\alpha', 'α'), (r'\beta',  'β'), (r'\gamma', 'γ'), (r'\delta', 'δ'),
    (r'\Delta', 'Δ'), (r'\phi',   'φ'), (r'\Phi',   'Φ'), (r'\omega', 'ω'),
    (r'\Omega', 'Ω'), (r'\mu',    'μ'), (r'\lambda','λ'), (r'\rho',   'ρ'),
    (r'\tau',   'τ'), (r'\pi',    'π'), (r'\Pi',    'Π'), (r'\epsilon','ε'),
    # Operators / symbols
    (r'\sum',   'Σ'), (r'\times', '×'), (r'\cdot',  '·'), (r'\approx','≈'),
    (r'\neq',   '≠'), (r'\leq',   '≤'), (r'\geq',   '≥'), (r'\pm',    '±'),
    (r'\infty', '∞'), (r'\rightarrow','→'), (r'\leftarrow','←'), (r'\Rightarrow','⇒'),
    # Degree
    (r'^\circ', '°'), (r'^{\circ}','°'), (r'\circ','°'), (r'\degree','°'),
]

# Pattern: equation lines — starts with math, not just prose containing math
# Matches things like:
#   P cos15° - 200(9.81) cos30° = 0
#   ΣF_x = 0
#   m_2 = m_1 sinθ / (1 - sinθ)
#   x̄ = 25.3 mm
#   ΣM_O = 0: P(0.12) + 0.05(9.81)(0.06) = 0
EQ_LINE_PATTERN = re.compile(
    r'^(?:\d+\.\s*|\*\s*|-\s*|>\s*)?'      # Optional list bullet
    r'(?:'
    r'[ΣΔαβγδθφωμλρτπεΑΒΓΔΘΦΩΛ∑]'          # starts with Greek
    r'|[A-Za-z][_\^]'                      # variable with subscript/superscript
    r'|[A-Z]\b'                            # single capital var (F, M, T)
    r'|[A-Za-z]+\s*(?:cos|sin|tan)'        # trig function
    r'|(?:cos|sin|tan|√)\s*[\d(]'          # trig/sqrt before number
    r')'
    r'.{0,120}=.{0,80}$',                  # has an equals sign
    re.IGNORECASE
)

# Inline math snippets — shorter patterns embedded in prose
INLINE_MATH_PATTERN = re.compile(
    r'(?<!\w)'                              # not preceded by word char
    r'(?:'
    r'[ΣΔαβγδθφωμλρτπεΑΒΓΩ][^\s,;.!?]{1,40}'  # Greek-led expression
    r'|[A-Za-z][_\^]\{?[A-Za-z0-9]+\}?'    # subscript/superscript var
    r'|(?:cos|sin|tan)[\s\d(°αβγθφ]{1,20}'  # trig
    r')'
    r'(?!\w)',
    re.IGNORECASE
)


def apply_symbols(text: str) -> str:
    """Replace LaTeX symbol commands with Unicode equivalents."""
    for latex_sym, uni in sorted(SYMBOLS, key=lambda x: -len(x[0])):
        text = text.replace(latex_sym, uni)
    return text


def convert_latex_expr(expr: str) -> str:
    """Convert a LaTeX math expression string to Unicode."""
    result = apply_symbols(expr)
    try:
        out = _latex.latex_to_text(result).strip()
        out = out.replace('\n', ' ')
        out = re.sub(r' {2,}', ' ', out)
        return out
    except Exception:
        result = result.replace('\n', ' ')
        return re.sub(r' {2,}', ' ', result)


def render_latex(text: str) -> str:
    r"""
    Pass 1: Convert explicit LaTeX delimiters.
    $$...$$ and \[...\]  →  ⟨ unicode ⟩   (display)
    $...$   and \(...\)  →  unicode         (inline)
    """
    def display(m):
        return f"  ⟨ {convert_latex_expr(m.group(1).strip())} ⟩"

    def inline(m):
        return convert_latex_expr(m.group(1).strip())

    text = re.sub(r'\$\$(.+?)\$\$', display, text, flags=re.DOTALL)
    text = re.sub(r'\\\[(.+?)\\\]',  display, text, flags=re.DOTALL)
    text = re.sub(r'\$(.+?)\$',      inline,  text, flags=re.DOTALL)
    text = re.sub(r'\\\((.+?)\\\)',  inline,  text, flags=re.DOTALL)

    # Also replace bare LaTeX commands outside math delimiters
    text = apply_symbols(text)
    return text


# ─── RICH TEXT BUILDER ────────────────────────────────────────────────────────

def build_rich_text(raw: str, render_math: bool) -> Group:
    """
    Build a Rich Text object from the LLM's answer.

    - If render_math=True:
        · Explicit $...$ / $$...$$ LaTeX → Unicode (pass 1)
        · Equation-like lines detected by pattern → Boxed in cyan
        · Inline math snippets in prose → highlighted cyan
    - Bold **...** always respected
    - Lists and structure preserved
    """
    # Pass 1: explicit LaTeX → unicode
    processed = render_latex(raw) if render_math else raw

    renderables = []
    current_text = Text()

    for line in processed.splitlines():
        stripped = line.strip()
        
        if not stripped:
            current_text.append("\n")
            continue

        # ── Display math lines  ⟨ ... ⟩ ────────────────────────────────
        if re.match(r'^\s*⟨.+⟩\s*$', line):
            if current_text.plain.strip():
                renderables.append(current_text)
            current_text = Text()

            eq_text = stripped.strip('⟨⟩ ')
            eq_panel = Panel(Text(eq_text, style="bold cyan", justify="center"),
                             border_style="cyan", box=box.ROUNDED, expand=False)
            renderables.append(Text("")) # padding top
            renderables.append(Align.center(eq_panel))
            renderables.append(Text("")) # padding bottom
            continue

        # ── Equation-like lines (pass 2 detection) ───────────────────────
        if render_math and EQ_LINE_PATTERN.search(stripped) and '=' in stripped:
            # Check it looks like math, not pure prose
            math_chars = sum(1 for c in stripped if c in '=+−-×÷/^_()[]ΣΔαβγδθφωμ°')
            is_prose = len(stripped) > 80 and (math_chars / max(1, len(stripped))) < 0.1
            if math_chars >= 2 and not is_prose:
                if current_text.plain.strip():
                    renderables.append(current_text)
                current_text = Text()

                eq_panel = Panel(Text(stripped, style="bold cyan", justify="center"),
                                 border_style="dim cyan", box=box.ROUNDED, expand=False)
                renderables.append(Text("")) # padding top
                renderables.append(Align.center(eq_panel))
                renderables.append(Text("")) # padding bottom
                continue

        # ── Normal prose line — handle bold and inline math ──────────────
        # Split by **bold**
        bold_parts = re.split(r'\*\*(.+?)\*\*', line)
        for bi, bp in enumerate(bold_parts):
            is_bold = (bi % 2 == 1)

            if is_bold:
                current_text.append(bp, style="bold white")
            else:
                # Split by ⟨inline display⟩
                disp_parts = re.split(r'(⟨.+?⟩)', bp)
                for di, dp in enumerate(disp_parts):
                    if dp.startswith('⟨') and dp.endswith('⟩'):
                        current_text.append(dp, style="bold cyan")
                    else:
                        if render_math:
                            # Pass 2 inline: highlight math snippets in prose
                            sub = re.split(r'(' + INLINE_MATH_PATTERN.pattern + r')',
                                          dp, flags=re.IGNORECASE)
                            for si, sp in enumerate(sub):
                                if si % 2 == 1:
                                    current_text.append(sp, style="cyan")
                                else:
                                    current_text.append(sp)
                        else:
                            current_text.append(dp)

        current_text.append("\n")

    if current_text.plain.strip():
        renderables.append(current_text)

    return Group(*renderables)


# ═══════════════════════════════════════════════════════════════════════════════
#  SETUP  (fix: get_or_create_collection)
# ═══════════════════════════════════════════════════════════════════════════════

def setup():
    global LLM_MODEL, LLM_PROVIDER
    
    console.print("\n╭─ [bold cyan]Select LLM Provider[/bold cyan]")
    console.print("│ 1: [green]OpenRouter (Cloud)[/green] - qwen/qwen3.6-plus-preview:free")
    console.print("│ 2: [yellow]Ollama (Local)[/yellow] - mightykatun/qwen2.5-math:1.5b")
    
    choice = Prompt.ask("╰─> [bold cyan]Choose an option[/bold cyan]", choices=["1", "2"], default="1")
    
    if choice == "1":
        LLM_PROVIDER = "OpenRouter"
        LLM_MODEL = "qwen/qwen3.6-plus-preview:free"
        base_url = "https://openrouter.ai/api/v1"
        api_key = os.environ.get("OPENROUTER_API_KEY", "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")
        if not api_key:
            console.print(Panel(
                "[bold red]OPENROUTER_API_KEY not set![/bold red]\n\n"
                "1. Get free key → https://openrouter.ai\n"
                "2. export OPENROUTER_API_KEY='sk-or-v1-...'",
                title="[red]Auth Error[/red]", border_style="red"
            ))
            sys.exit(1)
    else:
        LLM_PROVIDER = "Ollama"
        LLM_MODEL = "mightykatun/qwen2.5-math:1.5b"
        base_url = "http://localhost:11434/v1"
        api_key = "ollama"

    llm_client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    with console.status("[cyan]Connecting to vector DB...[/cyan]", spinner="dots"):
        try:
            db    = chromadb.PersistentClient(path=CHROMA_DIR)
            embed = embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=EMBED_MODEL)
            # FIX: use get_or_create so startup never crashes
            col   = db.get_or_create_collection(
                        name=COLLECTION,
                        embedding_function=embed,
                        metadata={"hnsw:space": "cosine"})

            count = col.count()
            if count == 0:
                console.print(Panel(
                    "[yellow]Vector DB is empty![/yellow]\n"
                    "Run [bold]python 1_extract.py[/bold] then [bold]python 2_embed.py[/bold] first.",
                    border_style="yellow"
                ))
                sys.exit(1)

        except Exception as e:
            console.print(Panel(
                f"[red]DB connection failed:[/red] {e}\n"
                "Make sure you've run 1_extract.py and 2_embed.py first.",
                title="[red]DB Error[/red]", border_style="red"
            ))
            sys.exit(1)

    return llm_client, col


# ═══════════════════════════════════════════════════════════════════════════════
#  SPLASH
# ═══════════════════════════════════════════════════════════════════════════════

def show_splash(col, current_model):
    console.clear()

    stats = Table(box=None, show_header=False, padding=(0, 2))
    stats.add_column(style="dim white")
    stats.add_column(style="bold cyan")
    stats.add_row("📚 Vector DB", f"{col.count()} chunks indexed")
    stats.add_row("🧠 LLM",       current_model)
    if LLM_PROVIDER == "Ollama":
        stats.add_row("🏠 Local",     "Ollama")
    else:
        stats.add_row("☁️  Cloud",     "OpenRouter (Free)")
    stats.add_row("🔍 Embeddings",EMBED_MODEL)
    stats.add_row("📐 Math",       "LaTeX + pattern detection ✓")

    cmds = Table(box=None, show_header=False, padding=(0, 2))
    cmds.add_column(style="bold yellow", min_width=26)
    cmds.add_column(style="dim white")
    cmds.add_row("help me solve 2/27",      "exact problem lookup")
    cmds.add_row("topic Equilibrium",       "browse all problems in topic")
    cmds.add_row("why? / explain step 3",   "follow-up questions")
    cmds.add_row("/model <name>",           "switch LLM model")
    cmds.add_row("/retry",                  "regenerate last response")
    cmds.add_row("/clear",                  "clear conversation history")
    cmds.add_row("/sources",                "toggle source panel")
    cmds.add_row("/context",                "toggle raw chunk text")
    cmds.add_row("/math",                   "toggle math highlighting")
    cmds.add_row("/quit",                   "exit")

    ASCII_LOGO = r"""[bold cyan]  ____ _____ _  _____ ___ ____    ____      _    ____ 
 / ___|_   _/ \|_   _|_ _/ ___|  |  _ \    / \  / ___|
 \___ \ | |/ _ \ | |  | | |      | |_) |  / _ \| |  _ 
  ___) || / ___ \| |  | | |___   |  _ <  / ___ \ |_| |
 |____/ |_/_/   \_\_| |___\____| |_| \_\/_/   \_\____|[/bold cyan]"""

    console.print()
    console.print(Align.center(ASCII_LOGO))
    console.print(Align.center("[dim cyan italic]Engineering Mechanics · 8th Edition · Meriam / Kraige / Bolton[/dim cyan italic]\n"))

    console.print(Columns([
        Panel(stats, title="[cyan]System Info[/cyan]",       border_style="cyan",   padding=(1, 2), box=box.ROUNDED),
        Panel(cmds,  title="[yellow]Commands[/yellow]", border_style="yellow", padding=(1, 2), box=box.ROUNDED),
    ], equal=True))
    console.print()


# ═══════════════════════════════════════════════════════════════════════════════
#  RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════════

PROB_PATTERN = re.compile(r'\b([A-Za-z]|\d+)\s*[/\-]\s*(\d+)\b')

def extract_problem_id(text: str) -> str | None:
    m = PROB_PATTERN.search(text)
    if m:
        ch, prob = m.group(1).strip().upper(), m.group(2).strip()
        if ch.isdigit() and int(ch) > 9:
            return None
        return f"ch{ch}_prob{prob}"
    return None


NO_OCR = ["[image —", "[native text only", "ocr not yet run", "use_vision_ocr"]

def has_real_content(t: str) -> bool:
    return not any(m in t.lower() for m in NO_OCR)


def retrieve(col, query, problem_id=None, topic=None):
    where, n = None, N_RESULTS
    if problem_id:
        where, n = {"problem_id": {"$eq": problem_id}}, 10
    elif topic:
        where = {"topic": {"$eq": topic}}

    res = col.query(
        query_texts=[query], n_results=n, where=where,
        include=["documents", "metadatas", "distances"]
    )
    chunks = [
        {"text": d, "meta": m, "similarity": round(1 - s, 3)}
        for d, m, s in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
    order = {"solution": 0, "question_bank": 1, "textbook": 2}
    chunks.sort(key=lambda x: order.get(x["meta"].get("source", ""), 3))
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
#  DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def show_sources(chunks, pid=None):
    t = Table(box=None, show_header=False,
              padding=(0, 1))
    
    for i, c in enumerate(chunks, 1):
        m            = c["meta"]
        src          = m.get("source", "?")
        style, label = SOURCE_STYLE.get(src, ("white", src.upper()))
        
        sim          = c["similarity"]
        sim_col      = "green" if sim > 0.7 else "yellow" if sim > 0.5 else "red"
        flag         = "[green]✓[/green]" if has_real_content(c["text"]) else "[yellow]⚠[/yellow]"
        
        t.add_row(
            f"[dim]{i}.[/dim]", 
            f"[{style}]{label}[/{style}]",
            f"[cyan]{m.get('problem_id','?')}[/cyan]",
            f"[dim]{m.get('topic','') or '—'}[/dim]",
            f"[{sim_col}]{sim:.2f}[/{sim_col}]", 
            flag
        )

    title = "\n[bold cyan]▶ Retrieved sources[/bold cyan]"
    if pid:
        title += f" [dim]· exact[/dim] [cyan]{pid}[/cyan]"
    
    group = Group(Text.from_markup(title), t, Text(""))
    console.print(group)


def show_answer(stream, turn: int, render_math: bool) -> str:
    raw_content = ""
    reasoning_content = ""
    
    initial_panel = Group(
        Text(f"╭─ 🤖 MechRAG · turn {turn}", style="bold magenta"),
        Text("╰─> ⠋ Thinking...", style="dim magenta")
    )
    import time
    start_time = time.time()
    ttft = None
    
    with Live(initial_panel, refresh_per_second=15, console=console) as live:
        for chunk in stream:
            # Stats part of the stream options if any
            if not chunk.choices:
                if hasattr(chunk, 'usage') and chunk.usage:
                    u = chunk.usage
                    stats_text = f"\n[dim white]Tokens: {u.prompt_tokens} prompt + {u.completion_tokens} completion. TTFT: {ttft:.2f}s.[/dim white]"
                    renderables = live.get_renderable().renderables if hasattr(live.get_renderable(), "renderables") else live.get_renderable()
                    if isinstance(renderables, tuple): 
                        renderables = list(renderables)
                    if isinstance(renderables, list):
                        renderables.append(Text.from_markup(stats_text))
                        live.update(Group(*renderables))
                continue
                
            if ttft is None:
                ttft = time.time() - start_time
                
            delta = chunk.choices[0].delta
            c = getattr(delta, "content", None)
            if c:
                raw_content += c
            r = getattr(delta, "reasoning_content", None)
            if r:
                reasoning_content += r
            
            display_think = reasoning_content
            display_ans = raw_content
            
            # If the model includes <think> tags natively in content, split it
            if "<think>" in raw_content:
                parts = raw_content.split("</think>")
                if len(parts) == 2:
                    t_part = parts[0].replace("<think>", "").strip()
                    display_think = (reasoning_content + "\n" + t_part).strip()
                    display_ans = parts[1].strip()
                else:
                    t_part = raw_content.replace("<think>", "").strip()
                    display_think = (reasoning_content + "\n" + t_part).strip()
                    display_ans = ""
            
            renderables = [Text(f"╭─ 🤖 MechRAG · turn {turn}", style="bold magenta")]
            if display_think:
                renderables.append(Text.assemble(
                    ("│ [Thinking Process]\n", "dim italic magenta"),
                    ("│ " + display_think.replace("\n", "\n│ "), "dim italic")
                ))
                renderables.append(Text("│", style="bold magenta"))
                
            if display_ans:
                ans_text = build_rich_text(display_ans, render_math)
                renderables.append(ans_text)
            elif not display_think:
                renderables.append(Text("╰─> ⠋ Thinking...", style="dim magenta"))
                
            # Keep the group without left padding so lines align with prompt, but maybe pad left slightly?
            # Actually, standard print is fine. We will pad left by 0 instead of 4 so it's flush.
            live.update(Group(*renderables))
            
    return raw_content


def show_user(query: str, pid: str = None):
    label = "👤 [bold yellow]You[/bold yellow]"
    if pid:
        label += f" [dim]· {pid}[/dim]"
    console.print(Padding(f"\n{label}\n[white]❯ {query}[/white]\n", (0, 4)))


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert tutor for Engineering Mechanics: Statics (8th Ed.).

You have access to [SOLUTION], [QUESTION BANK], and [TEXTBOOK] context.
Respond based on the type of user query:

1. NUMERICAL / CALCULATION PROBLEMS:
   - Goal: 1-sentence summary of what to find.
   - Approach: Brief state of theory used (e.g., FBD, sum of moments).
   - Verification: Provide a detailed step-by-step verification of math, outlining the derivation and double-checking all arithmetic operations to ensure zero calculation errors. (Do NOT use `<think>` tags).
   - Solution: Concise step-by-step mathematical derivation using bullet points. Explain *why* steps are taken. Ensure the math matches the verified steps.
   - Formatting: Wrap ONLY major milestone equations and the FINAL NUMERICAL ANSWER in display math (`$$ ... $$`). Do not put normal text inside `$$`.

2. CONCEPTUAL / THEORY QUESTIONS:
   - Direct, clear explanation avoiding conversational filler.
   - Use bullet points for readability.
   - DO NOT use `$$ ... $$` boxes for general theory statements or lists of equations. Rely on inline math (`$...$`) for variables (like $F_x$) or simple inline equations.
   - Keep answers strictly to the point without unnecessary concluding summaries.

GENERAL RULES:
- Never write huge walls of text; use whitespace and short paragraphs.
- Be concise. Skip pleasantries. Do not repeat the question.
- Rely on conversation history for follow-up questions."""


def generate(llm_client, query, chunks, history, current_model):
    sol_chunks = [c for c in chunks if c["meta"].get("source") == "solution"]
    if sol_chunks and not any(has_real_content(c["text"]) for c in sol_chunks):
        console.print(Panel(
            "[yellow]Solution OCR not run yet.[/yellow] Solving from question text.\n"
            "Run [bold]1_extract.py[/bold] with [cyan]USE_VISION_OCR=True[/cyan] for full solutions.",
            border_style="yellow", padding=(0, 2)
        ))

    ctx = []
    image_contents = []
    for c in chunks:
        src   = c["meta"].get("source","?").replace("_"," ").upper()
        pid   = c["meta"].get("problem_id","?")
        topic = c["meta"].get("topic","") or ""
        text  = c["text"] if has_real_content(c["text"]) else \
                "[Image only — OCR not run. Solve from question text.]"
        ctx.append(f"--- {src} | {pid} | {topic} ---\n{text}")

        img_path = c["meta"].get("image_path")
        if img_path and os.path.exists(img_path):
            try:
                with open(img_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
            except Exception as e:
                pass


    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history[-MAX_HISTORY:]:
        messages.append(turn)
        
    user_content = [{"type": "text", "text": f"Context:\n\n{chr(10).join(ctx)}\n\n---\nStudent: {query}"}]
    if image_contents:
        user_content.extend(image_contents)
        
    messages.append({
        "role":    "user",
        "content": user_content
    })

    try:
        resp = llm_client.chat.completions.create(
            model=current_model, messages=messages,
            temperature=0.2, max_tokens=2048,
            stream=True, stream_options={"include_usage": True},
        )
        return resp
    except Exception as e:
        # If model doesn't support images (e.g. 404 No endpoints found that support image input),
        # fallback to text only for the user turn context.
        if "image" in str(e).lower() or "404" in str(e):
            messages[-1]["content"] = user_content[:1] # Keep only the text part
            resp = llm_client.chat.completions.create(
                model=current_model, messages=messages,
                temperature=0.2, max_tokens=2048,
                stream=True, stream_options={"include_usage": True},
            )
            return resp
        else:
            raise e


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run(llm_client, col):
    current_model = LLM_MODEL
    show_splash(col, current_model)
    history, turn, show_src, render_math, show_ctx = [], 0, False, True, False
    active_pid = None

    while True:
        try:
            console.print("\n")
            user = Prompt.ask("[bold cyan]╭─ You[/bold cyan]\n[bold cyan]╰─>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user:
            continue

        if user.startswith("/"):
            cmd = user.lower().strip()
            if cmd in ("/quit", "/exit", "/q"):
                console.print("[dim]Goodbye![/dim]"); break
            elif cmd == "/clear":
                history, turn = [], 0
                active_pid = None
                show_splash(col, current_model)
                continue
            elif cmd.startswith("/model "):
                new_model = cmd.split(" ", 1)[1].strip()
                if new_model:
                    current_model = new_model
                    console.print(f"👉 [bold green]Model switched to:[/bold green] {current_model}")
                continue
            elif cmd == "/retry":
                if not history:
                    console.print("[yellow]No history to retry![/yellow]")
                    continue
                if history[-1]["role"] == "assistant":
                    history.pop() # remove last assistant
                if history and history[-1]["role"] == "user":
                    user = history.pop()["content"] # grab the last user message to resubmit
                    turn -= 1
                else:
                    console.print("[yellow]Could not find previous user prompt.[/yellow]")
                    continue
            elif cmd == "/sources":
                show_src = not show_src
                console.print(f"[dim]Source panel: {'[green]ON[/green]' if show_src else '[red]OFF[/red]'}[/dim]")
                continue
            elif cmd == "/context":
                show_ctx = not show_ctx
                console.print(f"[dim]Context raw text: {'[green]ON[/green]' if show_ctx else '[red]OFF[/red]'}[/dim]")
                continue
            elif cmd == "/math":
                render_math = not render_math
                console.print(f"[dim]Math highlighting: {'[green]ON[/green]' if render_math else '[red]OFF[/red]'}[/dim]")
                continue
            elif cmd == "/history":
                console.print(f"[dim]{len(history)//2} turns in context[/dim]")
                continue
            elif cmd in ("/help", "/?"):
                ht = Table(box=box.SIMPLE, show_header=False)
                ht.add_column(style="bold yellow", width=24)
                ht.add_column(style="dim white")
                for r in [("/clear","Clear history"), ("/sources","Toggle source panel"),
                           ("/math","Toggle math highlighting"), ("/history","Show turn count"),
                           ("/context","Toggle raw context text"),
                           ("/quit","Exit"), ("2/27","Direct problem lookup"),
                           ("topic X","Browse by topic"), ("why?","Follow-up question")]:
                    ht.add_row(*r)
                console.print(Panel(ht, title="[cyan]Help[/cyan]", border_style="dim"))
                continue
            else:
                console.print("[dim]Unknown command — type /help[/dim]")
                continue

        pid, topic = None, None
        if user.lower().startswith("topic "):
            topic = user[6:].strip()
            query = f"Show {topic} problems and solutions"
            active_pid = None
        else:
            extracted_pid = extract_problem_id(user)
            if extracted_pid:
                active_pid = extracted_pid
            pid = active_pid
            query = user

        # We skip show_user since the input prompt already displayed the text
        if pid:
            console.print(f"[dim]↳ Focused on problem: {pid}[/dim]\n")

        with console.status("[dim cyan]Searching knowledge base...[/dim cyan]", spinner="dots"):
            chunks = retrieve(col, query, problem_id=pid, topic=topic)

        if not chunks:
            console.print("[yellow]No results found. Try rephrasing.[/yellow]")
            continue

        if show_src:
            show_sources(chunks, pid)

        if show_ctx:
            ctx_renderables = [Text("-- Context Used --\n", style="dim")]
            for i, c in enumerate(chunks, 1):
                m = c["meta"]
                src = m.get("source", "?")
                style, label = SOURCE_STYLE.get(src, ("white", src.upper()))
                img_path = m.get("image_path")
                title = f"[{style}]{label}[/{style}] [dim]({m.get('problem_id','?')})[/dim]"
                
                ctx_renderables.append(Text.from_markup(title))
                ctx_renderables.append(Text("  " + c["text"].strip().replace("\n", "\n  ") + "\n", style="dim white"))
                if img_path and os.path.exists(img_path):
                    ctx_renderables.append(Text.from_markup(f"  [bold green]↳ 🖼️ Image attached: {img_path}[/bold green]\n"))
            
            console.print(Group(*ctx_renderables))
                
        turn += 1
        with console.status("[dim cyan]Evaluating prompt...[/dim cyan]", spinner="dots"):
            stream = generate(llm_client, query, chunks, history, current_model)

        answer = show_answer(stream, turn, render_math)

        history.append({"role": "user",      "content": user})
        history.append({"role": "assistant", "content": answer})
        if len(history) > MAX_HISTORY * 2:
            history = history[-(MAX_HISTORY * 2):]


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem",  help="e.g. '2/27'")
    parser.add_argument("--topic",    help="e.g. 'Force'")
    parser.add_argument("--question", help="Natural language question")
    args = parser.parse_args()

    llm_client, col = setup()

    if not any([args.problem, args.topic, args.question]):
        run(llm_client, col)
    else:
        pid   = extract_problem_id(args.problem) if args.problem else None
        topic = args.topic
        query = args.question or (
            f"Solve problem {args.problem} step by step" if args.problem
            else f"Show {topic} problems"
        )
        with console.status("[cyan]Searching...[/cyan]", spinner="dots"):
            chunks = retrieve(col, query, problem_id=pid, topic=topic)
        show_sources(chunks, pid)
        with console.status("[cyan]Generating...[/cyan]", spinner="dots"):
            answer = generate(llm_client, query, chunks, history=[])
        show_answer(answer, 1, render_math=True)
