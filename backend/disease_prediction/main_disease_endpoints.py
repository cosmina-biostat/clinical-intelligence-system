from fastapi import HTTPException
from pydantic import BaseModel


class DiseasePredictRequest(BaseModel):
    model_key: str          # "cardio" | "ms" | ... (from /disease/match or /disease/models)
    features: dict          # raw feature values, matching that model's feature_order


@app.get("/disease/match")
def disease_match(indication: str):
    """
    Given a protocol's indication string, return the auto-detected model
    (or null if no confident match -- the UI must then ask the user to pick).
    """
    card = DISEASE_REGISTRY.match_indication(indication)
    if card is None:
        return {"matched": False, "model_key": None,
                "candidates": [c.key for c in DISEASE_REGISTRY.available()]}
    return {
        "matched": True,
        "model_key": card.key,
        "display_name": card.display_name,
        "feature_order": card.feature_order,
        "notes": card.notes,
    }


@app.get("/disease/models")
def disease_models_list():
    """List every registered disease model, for a manual picker in the UI."""
    return {
        "models": [
            {"key": c.key, "display_name": c.display_name,
             "feature_order": c.feature_order, "notes": c.notes}
            for c in DISEASE_REGISTRY.available()
        ]
    }


@app.post("/disease/predict")
def disease_predict(req: DiseasePredictRequest):
    """
    Run the selected model on the given features. The model_key must come
    from a prior /disease/match or /disease/models call AND a human
    confirmation step in the UI -- this endpoint does not auto-select.
    """
    try:
        result = DISEASE_REGISTRY.predict(req.model_key, req.features)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except NotImplementedError as e:
        raise HTTPException(501, str(e))

    return {
        "model_key": result.model_key,
        "model_name": result.model_name,
        "label": result.label,
        "probability": result.probability,
    }
