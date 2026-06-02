# System architecture

## Three-layer design

```
Layer 1: Notebooks (training)
  notebooks/04_train_models.ipynb    → writes models/saved/*.pkl
  notebooks/05_anomaly_detection.ipynb → writes iso_forest_*.pkl
  notebooks/06_rag_pipeline.ipynb    → writes rag.index

Layer 2: Saved files (handoff)
  models/saved/
    xgboost_cardio.pkl
    lgbm_ms.pkl
    rf_melanoma.pkl
    iso_forest_cardio.pkl
    pipeline_cardio.pkl
    rag.index + rag_chunks.pkl

Layer 3: Streamlit dashboard (serving)
  app/app.py               → entry point
  app/pages/01_predict.py  → loads pkl, runs prediction + SHAP
  app/pages/02_anomaly.py  → loads iso_forest, shows flagged patients
  app/pages/03_models.py   → reads reports/model_comparison.csv
  app/pages/04_documents.py → loads FAISS index, calls Claude API
```

## Data flow for a prediction
1. User fills form in 01_predict.py
2. Patient data → pipeline.transform() → scaled array
3. Scaled array → model.predict_proba() → risk score
4. Scaled array → iso_forest.decision_function() → anomaly score
5. Scaled array → shap.TreeExplainer() → feature contributions
6. All three outputs rendered in Streamlit
