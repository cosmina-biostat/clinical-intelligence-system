def engineer_features(data: dict, flags: list, schema: dict) -> dict:

    # Get required fields from field_classification
    classification = schema.get("field_classification", {})
    required = classification.get("required", [])
    expected = classification.get("expected", [])

    # Completeness = how many required fields are actually filled
    if required:
        filled_required = sum(
            1 for field in required
            if data.get(field) is not None
        )
        completeness_score = round(filled_required / len(required), 2)
    else:
        completeness_score = 1.0

    # Field count (all extracted fields)
    field_count_total = len(data)

    # Flag counts (match checks.py types!)
    total_flags = len(flags)
    missing_required_count = sum(1 for f in flags if f["type"] == "missing" and f.get("severity") == "high")
    missing_optional_count = sum(1 for f in flags if f["type"] == "missing" and f.get("severity") == "low")
    out_of_range_count = sum(1 for f in flags if f["type"] == "out_of_range")
    plausibility_issues = sum(1 for f in flags if f["type"] == "plausibility")

    # Severity
    high_severity_flags = sum(1 for f in flags if f.get("severity") == "high")
    extraction_confidence = 1.0      # TODO: get from LLM

    # Critical = required fields missing
    critical_fields_missing = sum(
        1 for field in required
        if data.get(field) is None
    )

    return {
        "completeness_score": completeness_score,
        "missing_required_count": missing_required_count,
        "missing_optional_count": missing_optional_count,
        "out_of_range_count": out_of_range_count,
        "plausibility_issues": plausibility_issues,
        "total_flags": total_flags,
        "critical_fields_missing": critical_fields_missing,
        "high_severity_flags": high_severity_flags,
        "extraction_confidence": extraction_confidence,
        "field_count_total": field_count_total,
    }