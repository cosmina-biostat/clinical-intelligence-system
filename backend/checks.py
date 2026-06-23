def run_checks(data: dict, schema: dict) -> list:
    flags = []

    # --- Missing Check (by CDISC category) ---
    classification = schema.get("field_classification", {})
    required = classification.get("required", [])
    expected = classification.get("expected", [])

    for field in required:
        if data.get(field) is None:
            flags.append({
                "field": field, "type": "missing", "severity": "high",
                "message": f"required field {field} is missing"
            })

    for field in expected:
        if data.get(field) is None:
            flags.append({
                "field": field, "type": "missing", "severity": "low",
                "message": f"expected field {field} is missing"
            })

    # --- Range Check (unit-aware) ---
    ranges = schema.get("validation_ranges", {})

    for range_field, unit_ranges in ranges.items():
        # Find the patient's value (case-insensitive)
        value = None
        for data_field, data_value in data.items():
            if data_field.lower() == range_field.lower():
                value = data_value
                break

        if value is None:
            continue

        # Get the patient's unit for this field
        patient_unit = data.get(f"{range_field}_unit")

        # Select the matching unit's range
        limits = None
        if patient_unit and patient_unit in unit_ranges:
            limits = unit_ranges[patient_unit]
        elif len(unit_ranges) == 1:
            # Only one unit defined → use it
            limits = list(unit_ranges.values())[0]

        if limits is None:
            continue

        # Skip relative limits (xULN) — needs ULN reference (TODO)
        unit_key = patient_unit if patient_unit else list(unit_ranges.keys())[0]
        if "uln" in unit_key.lower():
            continue

        min_val = limits.get("min")
        max_val = limits.get("max")

        if min_val is not None and value < min_val:
            flags.append({
                "field": range_field, "type": "out_of_range", "severity": "medium",
                "message": f"{range_field}={value} below min {min_val} {unit_key}"
            })

        if max_val is not None and value > max_val:
            flags.append({
                "field": range_field, "type": "out_of_range", "severity": "medium",
                "message": f"{range_field}={value} above max {max_val} {unit_key}"
            })

    return flags