from protocol_parser import get_schema
from extractor import extract_data,save_record

# load schema
schema = get_schema("Prot_SAP_000.pdf")

# synthetic doctor's letter to test it
source_text = """
Patient ID: PT-00999
45-year-old female patient. ECOG performance status 1.
Diagnosis: Stage IIIC melanoma.

Laboratory values:
- Hemoglobin: 6.5 g/dL
- Platelets: 45,000/mcL
- ANC: 800/mcL
"""
# extract
result = extract_data(schema, source_text)

print(result)

save_record(result, "PT-00123.json")

#test checks.py
from checks import run_checks

flags = run_checks(result, schema)
print("FLAGS:", flags)

from feature_engineering import engineer_features

features = engineer_features(result, flags, schema)
print("FEATURES:", features)