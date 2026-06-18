import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


# ── Chunk data model ──────────────────────────────────────────────────────────
@dataclass
class Chunk:
    text: str
    section: str
    page: int
    chunk_index: int
    source: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # flatten for ChromaDB metadata (must be str/int/float/bool)
        meta = {
            "section": self.section,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "source": self.source,
        }
        meta.update({k: v for k, v in self.metadata.items()
                     if isinstance(v, (str, int, float, bool))})
        return {"text": self.text, "metadata": meta}


# ── Section detection ─────────────────────────────────────────────────────────
# Canonical clinical-protocol sections. Patterns are deliberately broad;
# real protocols vary, so we match on common headings (DE + EN).
SECTION_PATTERNS = {
    "eligibility": r"(inclusion|exclusion|eligibility|ein-?\s?und\s?ausschluss|einschlusskriterien|ausschlusskriterien)",
    "dosing":      r"(dosing|dose|dosage|administration|medication|study\s+drug|dosierung|verabreichung)",
    "endpoints":   r"(endpoint|outcome|efficacy|primary\s+endpoint|secondary\s+endpoint|zielparameter|endpunkt)",
    "safety":      r"(adverse\s+event|safety|toxicity|side\s+effect|sicherheit|nebenwirkung|unerw[üu]nschte)",
    "schedule":    r"(schedule\s+of\s+assessments|visit\s+schedule|study\s+procedures|ablaufplan|visitenplan)",
    "statistics":  r"(statistical\s+analysis|sample\s+size|statistik|fallzahl|analyse)",
    "objectives":  r"(objective|aim|purpose|hypothesis|zielsetzung|fragestellung)",
    "population":  r"(study\s+population|patient\s+population|subjects|studienpopulation|patienten)",
}

DEFAULT_SECTION = "general"


def detect_section(text: str, current: str) -> str:
    """
    Classify a page/heading into a section. Sticky: if no new section
    heading is found, the page inherits the current section (content
    continues across pages).

    Priority order matters: 'objectives' is checked before 'endpoints'
    because efficacy/outcome wording appears in both. We match against the
    first heading-like line, not the whole page top, to reduce false hits.
    """
    # Isolate the first non-empty line (the heading), lowercased
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    head = (lines[0] if lines else text[:120]).lower()

    # Explicit priority: more specific / heading-anchored sections first
    priority = [
        "objectives", "eligibility", "dosing", "endpoints",
        "safety", "schedule", "statistics", "population",
    ]
    for section in priority:
        if re.search(SECTION_PATTERNS[section], head, flags=re.IGNORECASE):
            return section
    return current


# ── PDF -> page texts ─────────────────────────────────────────────────────────
def read_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return list of (page_number, text). 1-indexed pages."""
    if PdfReader is None:
        raise ImportError("pypdf not installed. pip install pypdf")
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((i, text))
    return pages


# ── Hybrid chunking ───────────────────────────────────────────────────────────
def is_table_page(text: str) -> bool:
    """
    Heuristic: detect pages that are primarily tables.
    Handles both Unicode (≥/≤) and ASCII (>=/<=) variants.
    """
    text_lower = text.lower()
    
    # Unicode operators (many PDFs)
    unicode_ops = text.count("≥") + text.count("≤") >= 3
    
    # ASCII operators (KEYNOTE-054 style)
    ascii_ops = text.count(">=") + text.count("<=") >= 3
    
    # Table borders
    table_borders = text.count("|") >= 5
    
    # Clinical lab keywords + numbers (Table 2 pattern)
    lab_keywords = any(kw in text_lower for kw in [
        "anc", "platelet", "hemoglobin", "creatinine", 
        "bilirubin", "ast", "alt", "inr", "aptt"
    ])
    has_numbers = sum(1 for w in text.split() if w.replace(",","").replace(".","").isdigit()) >= 5
    lab_table = lab_keywords and has_numbers
    
    # Tab-based tables
    tab_table = text.count("\t") >= 10 and lab_keywords
    
    return unicode_ops or ascii_ops or table_borders or lab_table or tab_table


def chunk_protocol(
    pdf_path: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    char_size    = chunk_size * 4
    char_overlap = chunk_overlap * 4

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=char_size,
        chunk_overlap=char_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    source = pdf_path.name
    pages = read_pdf_pages(pdf_path)

    # Stage 1: section detection (sticky)
    current_section = DEFAULT_SECTION
    page_sections = []
    for page_no, text in pages:
        current_section = detect_section(text, current_section)
        page_sections.append((page_no, current_section, text))

    # Stage 2: hybrid chunking
    chunks: list[Chunk] = []
    global_idx = 0
    for page_no, section, text in page_sections:
        if not text.strip():
            continue

        if is_table_page(text):
            # Keep entire page as ONE chunk — don't split tables!
            chunks.append(Chunk(
                text=text.strip(),
                section=section,
                page=page_no,
                chunk_index=global_idx,
                source=source,
                metadata={"is_table": True},
            ))
            global_idx += 1
        else:
            # Normal recursive split for prose
            for piece in splitter.split_text(text):
                piece = piece.strip()
                if len(piece) < 20:
                    continue
                chunks.append(Chunk(
                    text=piece,
                    section=section,
                    page=page_no,
                    chunk_index=global_idx,
                    source=source,
                ))
                global_idx += 1

    return chunks

# ── Summary ───────────────────────────────────────────────────────────────────
def summarise(chunks: list[Chunk]) -> None:
    from collections import Counter
    by_section = Counter(c.section for c in chunks)
    total_chars = sum(len(c.text) for c in chunks)
    print(f"  Total chunks : {len(chunks)}")
    print(f"  Avg chars    : {total_chars // max(len(chunks),1)}")
    print(f"  Pages covered: {len(set(c.page for c in chunks))}")
    print(f"  Sections:")
    for section, count in by_section.most_common():
        print(f"    {section:<14} {count:>4} chunks")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python rag_chunker.py <protocol.pdf>")
        raise SystemExit(1)

    pdf = Path(sys.argv[1])
    print(f"Chunking {pdf.name} ...")
    chunks = chunk_protocol(pdf)
    summarise(chunks)

    # Preview first chunk per section
    seen = set()
    print("\n  Sample chunk per section:")
    for c in chunks:
        if c.section not in seen:
            seen.add(c.section)
            preview = c.text[:120].replace("\n", " ")
            print(f"    [{c.section}, p.{c.page}] {preview}...")