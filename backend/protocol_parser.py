from pypdf import PdfReader
from anthropic import Anthropic
from dotenv import load_dotenv
from pathlib import Path
from backend.field_names import STANDARD_FIELD_NAMES
from backend.rag_chunker import chunk_protocol
from backend.rag_store import ProtocolVectorStore, LocalEmbedder
import json
import os
import time

load_dotenv(Path(__file__).parent / ".env")


def get_or_build_index(pdf_path: str) -> ProtocolVectorStore:
    """
    Builds ChromaDB index from PDF (once).
    Persistent — subsequent calls load from disk, no re-embedding.
    """
    store = ProtocolVectorStore(
        persist_dir=str(Path(pdf_path).parent / "chroma_db"),
        embedder=LocalEmbedder(),
    )

    source = Path(pdf_path).name
    if store.is_indexed(source):
        print(f"  RAG index loaded from disk: {source}")
        return store

    print(f"  Building RAG index for {source} ...")
    chunks = chunk_protocol(Path(pdf_path))
    store.build_index([c.to_dict() for c in chunks])
    return store


def retrieve_context(store: ProtocolVectorStore, source: str) -> str:
    """
    Disease-agnostic retrieval. Queries target the STRUCTURE common to every
    clinical protocol (eligibility, dosing, endpoints, labs ...), not the
    terminology of one indication. This works across cardiology, MI, MS,
    oncology, etc. -- no melanoma-specific terms.

    Each query pairs an optional section filter with generic search text.
    The section filter narrows the search before ranking; the text finds
    the right passage within that section regardless of disease area.
    """
    queries = [
        # General study info (indication-neutral)
        (None, "study name title protocol number version sponsor"),
        (None, "study design randomized phase double blind placebo controlled"),
        (None, "indication disease condition primary objective purpose"),
        (None, "laboratory values eligibility criteria thresholds table"),

        # Eligibility (generic clinical wording)
        ("eligibility", "inclusion criteria age sex performance status diagnosis"),
        ("eligibility", "exclusion criteria prior therapy comorbidity contraindication"),
        ("eligibility", "laboratory hematology chemistry blood values required"),
        ("eligibility", "renal hepatic function creatinine bilirubin enzymes coagulation"),

        # Dosing + Medication (no drug name hardcoded)
        ("dosing", "study drug dose dosage administration route frequency"),
        ("dosing", "prohibited medications concomitant restricted contraindicated"),
        ("dosing", "allowed permitted concomitant medication exceptions"),

        # Safety + Schedule
        ("safety", "adverse events grading stopping rules discontinuation criteria"),
        ("schedule", "visit schedule assessments timepoints follow-up intervals"),

        # Endpoints + Population
        ("endpoints", "primary endpoint secondary endpoint outcome measure"),
        ("population", "study population patients subjects sample size"),

        # Vitals / common cardiovascular + neuro measures (broad, not exclusive)
        (None, "blood pressure heart rate ejection fraction ECG troponin"),
        (None, "EDSS MRI lesions neurological score disability assessment"),
    ]

    seen = set()
    chunks = []
    for section, query in queries:
        hits = store.query(query, top_k=5, section=section, source=source)
        for hit in hits:
            text = hit["text"]
            if text not in seen:
                seen.add(text)
                meta = hit["metadata"]
                chunks.append(
                    f"[Section: {meta.get('section')}, Page: {meta.get('page')}]\n{text}"
                )

    context = "\n\n---\n\n".join(chunks)
    print(f"  RAG: {len(chunks)} unique chunks retrieved "
          f"(~{len(context)//4} tokens sent to LLM instead of the full PDF)")
    return context


def parse_protocol(pdf_path: str) -> dict:
    """Reads a clinical study protocol PDF and returns a StudySchema dict."""

    # 1. Build/load RAG index
    store = get_or_build_index(pdf_path)
    source = Path(pdf_path).name

    # 2. Retrieve relevant chunks (replaces full PDF text)
    text = retrieve_context(store, source)

    # 3. Prompt
    prompt = f"""You are a clinical data manager who strictly follows instructions.
Extract the following fields:
- study_name
- indication
- study_type
- protocol_version
- visit_structure
- field_definitions
- field_classification
- validation_rules
- validation_ranges
- data_types
- eligibility_criteria
- medication_guidelines

When creating field_classification and validation_ranges, use these STANDARD field names where the field matches (short snake_case, no units):
{STANDARD_FIELD_NAMES}

Rules:
- Only information explicitly stated in the text
- Respond ONLY with JSON, no text before or after
- If a value is not clearly stated in the text, return null — never invent
- For field_classification: classify fields into three CDISC core categories:
  - "required": core identifying fields that must always be present (patient_id, age, sex)
  - "expected": clinically important fields that may be filled across multiple documents (lab values, vitals)
  - "permissible": optional fields
- For validation_ranges: structure each field BY UNIT. Use the unit as a key (e.g. "g/dL", "mmol/L", "/mcL"). If the protocol gives the same field in multiple units, include each unit separately. Use null where no limit is specified. For relative limits like "2.5 xULN", use "xULN" as the unit key.
- For validation_ranges keys: use the STANDARD field names above where applicable; for other fields create a clean snake_case name (lowercase, no units)
- For eligibility_criteria: extract inclusion and exclusion criteria as two separate lists, each item as a short clear statement
- For medication_guidelines: extract study medication, prohibited medications/treatments, and allowed exceptions as structured lists

Respond ONLY with JSON in this format:
{{
    "study_name": "...",
    "indication": "...",
    "study_type": "...",
    "protocol_version": "...",
    "visit_structure": "...",
    "field_definitions": "...",
    "field_classification": {{
        "required": ["patient_id", "age", "sex"],
        "expected": ["hemoglobin", "platelets", "anc"],
        "permissible": ["crp", "ldh"]
    }},
    "validation_rules": "...",
    "validation_ranges": {{"hemoglobin": {{"g/dL": {{"min": 9, "max": null}}, "mmol/L": {{"min": 5.6, "max": null}}}}, "creatinine": {{"mg/dL": {{"min": null, "max": 1.5}}}}}},
    "data_types": "...",
    "eligibility_criteria": {{
        "inclusion": ["Age >= 18 years", "Confirmed primary diagnosis per protocol"],
        "exclusion": ["Prior investigational therapy", "Significant comorbidity per protocol"]
    }},
    "medication_guidelines": {{
        "study_medication": "Study drug, dose and frequency as specified in protocol",
        "prohibited": ["Medications contraindicated per protocol"],
        "allowed": ["Concomitant medication within protocol limits"]
    }}
}}"""

    # 4. Build content (RAG chunks instead of full PDF)
    content = f"""
<prompt>
{prompt}
</prompt>

<document>
{text}
</document>
"""

    # 5. API call
    client = Anthropic()

    text_response = ""
    for attempt in range(3):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8000,
            temperature=0,
            messages=[{"role": "user", "content": content}],
        )

        text_response = ""
        for block in response.content:
            if block.type == "text":
                text_response += block.text

        print(f"Attempt {attempt+1}: got {len(text_response)} chars")

        if text_response.strip():
            break
        time.sleep(3)

    # 6. Clean markdown backticks if present
    text_response = text_response.strip()
    if text_response.startswith("```"):
        text_response = text_response.split("```")[1]
        if text_response.startswith("json"):
            text_response = text_response[4:]
        text_response = text_response.strip()

    # 7. Parse JSON
    result = json.loads(text_response)

    return result


def get_schema(pdf_path: str, cache_path: str = None) -> dict:
    """Returns the StudySchema. Uses cached JSON if available (saves tokens)."""

    if cache_path is None:
        cache_path = Path(pdf_path).stem + "_schema.json"

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            print(f"Schema loaded from cache: {cache_path}")
            return json.load(f)

    schema = parse_protocol(pdf_path)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
        print(f"Schema created and cached: {cache_path}")

    return schema


def get_rag_store(pdf_path: str) -> ProtocolVectorStore:
    """
    Returns the RAG store for a protocol.
    Used by Dashboard chatbox — same index, no re-embedding.
    """
    return get_or_build_index(pdf_path)