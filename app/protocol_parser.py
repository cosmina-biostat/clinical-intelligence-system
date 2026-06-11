from pypdf import PdfReader
from anthropic import Anthropic
from dotenv import load_dotenv
from pathlib import Path
from field_names import STANDARD_FIELD_NAMES
import json
import os
import time

# Load .env from the same folder as this file
load_dotenv(Path(__file__).parent / ".env")


def parse_protocol(pdf_path: str) -> dict:
    """Reads a clinical study protocol PDF and returns a StudySchema dict."""

    # 1. Extract PDF text
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # 2. Prompt
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
    "validation_ranges": {{"hemoglobin": {{"g/dL": {{"min": 9, "max": null}}, "mmol/L": {{"min": 5.6, "max": null}}}}, "anc": {{"/mcL": {{"min": 1500, "max": null}}}}}},
    "data_types": "..."
}}"""

    # 3. Build content
    content = f"""
<prompt>
{prompt}
</prompt>

<document>
{text}
</document>
"""

    # 4. API call (with retry for empty reseller responses)
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

    # 5. Clean markdown backticks if present
    text_response = text_response.strip()
    if text_response.startswith("```"):
        text_response = text_response.split("```")[1]
        if text_response.startswith("json"):
            text_response = text_response[4:]
        text_response = text_response.strip()

    # 6. Parse JSON
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