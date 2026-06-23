from pathlib import Path
from backend.protocol_parser import get_schema

# Pfad relativ zu DIESER Datei, egal von wo gestartet wird
pdf = Path(__file__).parent / "protocols" / "Prot_SAP_000.pdf"
schema = get_schema(str(pdf))
print(schema)