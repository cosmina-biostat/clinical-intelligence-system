from protocol_parser import get_schema
from extractor import process_documents, save_record

schema = get_schema("Prot_SAP_000.pdf")

# merge test for doc 1 and 2
# pat 1 Document 1
doc1 = """
Patient ID: PT-001
Date of visit: 10 June 2026
54-year-old male patient. ECOG performance status 0.
Diagnosis: Stage IIIB melanoma, completely resected (R0).
BRAF mutation status: negative.
PD-L1 expression: positive (TPS 15%).
"""

# pat 1 Document 2
doc2 = """
Patient ID: PT-001
Laboratory values:
- Hemoglobin: 6.5 g/dL
- Platelets: 45,000/mcL
- ANC: 800/mcL
- Creatinine: 0.9 mg/dL
- AST: 28 U/L, ALT: 31 U/L
"""

# pat 2
doc3 = """
Patient ID: PT-002
45-year-old female. ECOG 1.
Stage IIIC melanoma.
Hemoglobin: 13.8 g/dL
Platelets: 245,000/mcL
ANC: 4,200/mcL
"""

records = process_documents(schema, [doc1, doc2, doc3])

for record in records:
    print(f"\n--- Patient: {record['data'].get('patient_id')} ---")
    print(f"FLAGS: {len(record['flags'])} total")
    print(f"FEATURES: {record['features']}")
    save_record(record, f"{record['data'].get('patient_id')}.json")