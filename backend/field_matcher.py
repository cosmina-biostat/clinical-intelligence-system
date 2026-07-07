import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from backend.field_names import STANDARD_FIELD_NAMES

ALIAS_MAP_PATH = Path(__file__).parent / "field_aliases.json"
SIMILARITY_THRESHOLD = 0.75  # minimum cosine similarity for a match

class FieldMatcher:
    def __init__(self, model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"):
        self.standard_fields = STANDARD_FIELD_NAMES
        self.alias_map = self._load_aliases()
        self.model = None  # lazy load (only when needed)
        self.model_name = model
        self._field_embeddings = None
    
    def _load_aliases(self) -> dict:
        if ALIAS_MAP_PATH.exists():
            with open(ALIAS_MAP_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def _save_aliases(self):
        with open(ALIAS_MAP_PATH, "w", encoding="utf-8") as f:    
            json.dump(self.alias_map, f, indent=2, ensure_ascii=False)
    
    def _load_model(self):
        """Lazy load — only when vector matching needed."""
        if self.model is None:
            print("  Loading field matcher model...")
            self.model = SentenceTransformer(self.model_name)
            self._field_embeddings = self.model.encode(
                self.standard_fields, convert_to_tensor=True
            )
    
    def match(self, field_name: str, auto_save: bool = True) -> tuple[str, str, float]:
        """
        Match a field name to a standard field.
        Returns: (matched_field, method, confidence)
        method: "exact" | "alias" | "vector" | "unknown"
        """
        normalized = field_name.lower().strip()
        
        # Stage 1: Exact match
        if normalized in [f.lower() for f in self.standard_fields]:
            return normalized, "exact", 1.0
        
        # Stage 2: Alias map
        if normalized in self.alias_map:
            return self.alias_map[normalized], "alias", 1.0
        
        # Stage 3: Vector similarity
        self._load_model()
        query_emb = self.model.encode(field_name, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self._field_embeddings)[0]
        best_idx = scores.argmax().item()
        best_score = scores[best_idx].item()
        best_field = self.standard_fields[best_idx]
        
        if best_score >= SIMILARITY_THRESHOLD:
            # Save to alias map for future runs
            if auto_save:
                self.alias_map[normalized] = best_field
                self._save_aliases()
                print(f"  Field matched: '{field_name}' → '{best_field}' "
                      f"(score={best_score:.2f}) → saved to aliases")
            return best_field, "vector", best_score
        
        # No match found
        return field_name, "unknown", best_score
    
    def match_record(self, record: dict) -> dict:
        """
        Match all field names in a record.
        Returns new record with standardized field names.
        """
        matched = {}
        for field, value in record.items():
            if field.endswith("_unit"):  # skip unit fields
                continue
            std_field, method, score = self.match(field)
            matched[std_field] = value
            # Preserve unit field with new name
            unit_key = f"{field}_unit"
            if unit_key in record:
                matched[f"{std_field}_unit"] = record[unit_key]
        return matched