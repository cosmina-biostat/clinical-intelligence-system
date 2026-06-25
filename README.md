# Clinical Data Intelligence & Monitoring System
> Capstone project — Data Science & AI Bootcamp

A unified platform combining **ML risk prediction**, **explainability (SHAP)**, **anomaly detection**, and **clinical document Q&A** — all through a single Streamlit dashboard requiring zero technical knowledge to use.

---

## What it does

| Page | Capability | How it works |
|------|-----------|--------------|
| 🫀 **Risk Prediction** | Enter patient vitals → CVD risk score in <1 second | LightGBM trained on 70,000 records |
| 🔬 **SHAP Explanation** | Plain-English breakdown of what drove the prediction | TreeExplainer — per-patient feature contributions |
| 🔍 **Anomaly Monitor** | Flags patients with unusual profiles vs the cohort | Isolation Forest, 2% contamination rate |
| 📊 **Model Comparison** | Side-by-side AUC / F1 for 4 classifiers | LR · RF · XGBoost · LightGBM |
| 📄 **Document Q&A** | Upload a clinical PDF → ask questions in plain English | ChromaDB + Claude Haiku (RAG pipeline) |

---

## Results — Cardiovascular dataset (68,595 patients, 20% test set)

| Classifier | ROC-AUC | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| **LightGBM** ⭐ | **0.8065** | **0.7198** | 0.7559 | 0.6870 | 0.7354 |
| XGBoost | 0.7961 | 0.7167 | 0.7560 | 0.6813 | 0.7336 |
| Logistic Regression | 0.7950 | 0.7092 | 0.7602 | 0.6647 | 0.7304 |
| Random Forest | 0.7759 | 0.7088 | 0.7211 | 0.6969 | 0.7167 |

LightGBM is the best-performing model and is used for live predictions in the dashboard.

---

## Quickstart — run the app locally

### 1. Clone the repo
```bash
git clone https://github.com/cosmina-biostat/clinical-intelligence-system.git
cd clinical-intelligence-system
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Mac / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Anthropic API key
```bash
cp .env.example .env
```
Open `.env` and replace the placeholder with your real key:
```
ANTHROPIC_API_KEY=sk-ant-...
```
> The Document Q&A page requires this key. All other pages (Predict, Anomaly, Model Comparison) work without it.

### 5. Add the cardiovascular dataset
Download [`cardio_train.csv`](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) from Kaggle and place it at:
```
data/raw/cardio_train.csv
```

### 6. Launch the dashboard
```bash
streamlit run app/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

> **First load:** Pages 01–03 load pre-trained models from `models/saved/` — no retraining needed. The app is ready in seconds.

---

## Repository structure

```
clinical-intelligence-system/
│
├── app/
│   ├── app.py                  # Landing page + navigation
│   ├── cardio_app.py           # Standalone self-contained demo (alternative entry)
│   └── pages/
│       ├── 01_predict.py       # Risk prediction + SHAP
│       ├── 02_anomaly.py       # Isolation Forest monitor
│       ├── 03_models.py        # 4-classifier comparison + radar chart
│       └── 04_documents.py     # RAG Q&A + protocol schema + record classifier
│
├── backend/                    # Clinical protocol intelligence (Ziya)
│   ├── rag_chunker.py          # Section-aware PDF chunking
│   ├── rag_store.py            # ChromaDB vector store
│   ├── protocol_parser.py      # Claude-powered schema extraction
│   ├── inference.py            # Review classifier + quality regressor
│   ├── models/                 # Pre-trained .pkl classifiers
│   └── protocols/              # PDFs + ChromaDB index
│
├── src/
│   ├── data/                   # Loaders, preprocessor
│   ├── models/                 # train.py, evaluate.py, explain.py
│   ├── rag/                    # pdf_parser, embedder, retriever (FAISS fallback)
│   ├── anomaly/                # detector.py
│   └── utils/                  # config.py, logger.py
│
├── notebooks/
│   ├── 00_eda_heart.ipynb
│   ├── 01_eda_cardiovascular.ipynb     # Main EDA — 70k cardio records
│   ├── 02_eda_ms_cis.ipynb
│   ├── 03_eda_melanoma.ipynb
│   ├── 04_train_models.ipynb
│   ├── 05_anomaly_detection.ipynb
│   ├── 06_rag_pipeline.ipynb
│   ├── 07_model_comparison.ipynb
│   └── cardio_full_pipeline.ipynb      # Full pipeline in one notebook
│
├── models/saved/               # Pre-trained .pkl files (committed, except RF)
├── reports/
│   └── model_comparison.csv    # AUC / F1 / precision / recall for all classifiers
├── data/
│   ├── raw/                    # gitignored — add cardio_train.csv here
│   └── processed/              # gitignored — cleaned CSVs generated at runtime
│
├── tests/                      # pytest suite
├── docs/                       # setup.md, architecture.md, team_workflow.md
├── Dockerfile                  # Container build
├── requirements.txt
└── .env.example                # API key template
```

---

## Running the notebooks (optional — models already saved)

The notebooks document the full analysis. You do **not** need to run them to use the app — the trained `.pkl` files are already committed.

If you want to explore the analysis:
```bash
pip install jupyterlab
jupyter lab
```
Run in order: `01_eda_cardiovascular.ipynb` → `04_train_models.ipynb` → `07_model_comparison.ipynb`

---

## Docker (optional)

```bash
docker build -t clinical-intelligence .
docker run -e ANTHROPIC_API_KEY=sk-ant-... -p 8501:8501 clinical-intelligence
```

---

## Tech stack

| Layer | Libraries |
|---|---|
| ML | scikit-learn · XGBoost · LightGBM · imbalanced-learn |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| RAG & LLM | Anthropic Claude · ChromaDB · sentence-transformers · LangChain |
| PDF parsing | pdfplumber · PyMuPDF · pypdf |
| Dashboard | Streamlit · Plotly |
| Data | pandas · numpy |
| Testing | pytest |
| Packaging | Docker · requirements.txt |

---

## Team
| Person | Contributions |
|--------|--------------|
| Cosmina | Cardiovascular EDA · LightGBM tuning · SHAP · Streamlit dashboard · deployment |
| Ziya | Clinical protocol backend · ChromaDB RAG · review classifier · quality regressor |
| Manish | Additional datasets · model comparison · notebooks |
