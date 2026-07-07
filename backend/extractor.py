from anthropic import Anthropic
from dotenv import load_dotenv
from pathlib import Path
from backend.field_names import STANDARD_FIELD_NAMES
from backend.checks import run_checks
from backend.feature_engineering import engineer_features
import json
import os
from backend.field_matcher import FieldMatcher

load_dotenv(Path(__file__).parent / ".env")

_matcher = FieldMatcher()

def extract_data(schema: dict, source_text: str) -> dict:

    # 1. Prompt
    # Build the list of fields to extract from field_classification.
    # Includes permissible fields too -- otherwise fields like disease-risk
    # covariates (e.g. bmi, cholesterol, smoking) that a protocol lists as
    # "permissible" are never even mentioned to the LLM and can never be
    # extracted, even when they're clearly present in the source text.
    classification = schema.get("field_classification", {})
    fields_to_extract = (
        classification.get("required", [])
        + classification.get("expected", [])
        + classification.get("permissible", [])
    )

    # 1. Prompt
    prompt = f"""You are a clinical data extractor who strictly follows instructions.

Extract values from the source document for these fields:
{fields_to_extract}

Use these STANDARD field names where the field matches (short snake_case, no units):
{STANDARD_FIELD_NAMES}

Expected data types:
{schema["data_types"]}

Rules:
- The source document may be in any language (German, English, etc.) — map all values to the English standard field names above
- Only extract values explicitly stated in the text
- Always include patient_id to link the data
- Separate values from units: use "field" for the numeric value and "field_unit" for the unit
- Field names must NOT contain units (use "hemoglobin", not "hemoglobin_g_dL")
- For fields not in the standard list, use a clean snake_case name (lowercase, no units)
- Convert decimal commas to points (13,8 → 13.8)
- If a field is not found, return null — never invent
- Respond ONLY with JSON, no text before or after

Respond ONLY with JSON in this format:
{{
    "patient_id": "...",
    "hemoglobin": 13.8,
    "hemoglobin_unit": "g/dL"
}}"""

    # 2. API call
    client = Anthropic()

    content = f"""
<prompt>
{prompt}
</prompt>

<document>
{source_text}
</document>
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        temperature=0,
        messages=[{"role": "user", "content": content}],
    )

    # 3. Extract text from response (skip ThinkingBlock)
    text_response = ""
    for block in response.content:
        if block.type == "text":
            text_response += block.text

    # 4. Clean markdown backticks if present
    text_response = text_response.strip()
    if text_response.startswith("```"):
        text_response = text_response.split("```")[1]
        if text_response.startswith("json"):
            text_response = text_response[4:]
        text_response = text_response.strip()

    # 5. Parse JSON
    result = json.loads(text_response)
    
    # 6. Match field names → standard names (Hybrid: exact → alias → vector)
    result = _matcher.match_record(result)
    return result

def save_record(record: dict, filename: str):
    """Saves an extracted record as JSON."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
        print(f"Record saved: {filename}")

def merge_records(existing: dict, new: dict) -> dict:
    for field, value in new.items():
        if existing.get(field) is None and value is not None:
            existing[field] = value
    return existing

def process_documents(schema: dict, documents: list) -> list:
    patient_records = {}
    
    for source_text in documents:
        result = extract_data(schema, source_text)
        pid = result.get("patient_id")
        
        if pid not in patient_records:
            patient_records[pid] = result
        else:
            patient_records[pid] = merge_records(patient_records[pid], result)
    
    complete_records = []
    for pid, merged_data in patient_records.items():
        flags = run_checks(merged_data, schema)
        features = engineer_features(merged_data, flags, schema)
        complete_records.append({
            "data": merged_data,
            "flags": flags,
            "features": features
        })
    
    return complete_records