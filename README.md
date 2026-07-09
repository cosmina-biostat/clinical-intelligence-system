# CDIM System — Clinical Data Intelligence & Monitoring

> Capstone project — Data Science & AI Bootcamp

A production-style clinical ML platform combining cardiovascular risk prediction, explainability (SHAP), anomaly detection, and LLM-powered document intelligence — all in a single Streamlit dashboard.

---

## What the system does

### 🫀 Patient Risk & Anomaly Checker
- **Cardiovascular risk prediction** — LightGBM trained on 68,595 patient records (ROC-AUC 0.8065)
- **Per-patient SHAP explanations** — bar chart showing which features drove the prediction up or down
- **Isolation Forest anomaly detection** — flags patient profiles that are statistically unusual relative to the training cohort
- Inputs: age, sex, height, weight, BP, cholesterol, glucose, lifestyle factors

### 📄 Document Intelligence (5-tab layout)
| Tab | What it does |
|-----|-------------|
| **Protocol & Chat** | Upload a clinical PDF → Claude extracts the study schema; ask free-text questions grounded in the document with source citations |
| **Extraction** | Paste or upload patient letters → structured fields extracted per protocol schema, traffic-light review verdict (Clean / Review / Block) |
| **Structured** | Filterable table with coloured quality bars, risk predictions, and per-patient detail cards |
| **Monitoring** | Aggregate review status overview (Clean / Review / Block counts) |
| **Insights** | Donut charts for demographics, risk tiers, and data quality distribution |

---

## Model results — Cardiovascular dataset (68,595 patients, 20% test set)

| Classifier | ROC-AUC | F1 | Accuracy |
|---|---|---|---|
| **LightGBM** ⭐ | **0.8065** | **0.7198** | **0.7354** |
| XGBoost | 0.7961 | 0.7167 | 0.7336 |
| Logistic Regression | 0.7950 | 0.7092 | 0.7304 |
| Random Forest | 0.7759 | 0.7088 | 0.7167 |

LightGBM is the best-performing model and is used for live predictions.

---

## Architecture

```
clinical-intelligence-system/
├── app/
│   ├── app.py                    # Landing page
│   └── pages/
│       ├── 01_predict.py         # Risk prediction + SHAP + anomaly
│       └── 02_documents.py       # Document intelligence (5-tab layout)
├── backend/
│   ├── protocol_parser.py        # PDF → schema (Claude API + RAG)
│   ├── rag_store.py              # ChromaDB vector store + local embedder
│   ├── rag_chunker.py            # Section-aware PDF chunking
│   ├── extractor.py              # Patient letter → structured fields
│   ├── inference.py              # Review classification + quality scoring
│   ├── feature_engineering.py    # Feature computation for review model
│   ├── checks.py                 # Validation rules engine
│   ├── models/                   # Pre-trained review + quality .pkl files
│   └── disease_prediction/
│       ├── disease_models.py     # DiseaseModelRegistry, cardio feature resolver
│       └── models/               # LightGBM pipeline + scaler
├── models/saved/                 # Cardio LightGBM, scaler, Isolation Forest
├── notebooks/                    # EDA and full pipeline notebooks
└── scripts/training_recipes/     # Reproducible training scripts
```

**Stack:** LightGBM · SHAP · Isolation Forest · ChromaDB · Claude API · Streamlit · Plotly · scikit-learn · Python 3.11

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/cosmina-biostat/clinical-intelligence-system.git
cd clinical-intelligence-system
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **macOS with pyenv:** If you get a segfault on startup, your Python was compiled without lzma. Fix with:
> ```bash
> brew install xz && pyenv install 3.11.3 --force
> ```

### 4. Set your Anthropic API key

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...your-key-here...
```

Get a key at [console.anthropic.com](https://console.anthropic.com).  
The Protocol & Chat tab requires this key. All other pages work without it.

### 5. Run the app

```bash
python -m streamlit run app/app.py
```

> **macOS only — if you get a segfault** (sentence-transformers + forked processes):
> ```bash
> TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 python -m streamlit run app/app.py
> ```

Open **http://localhost:8501** in your browser.

---

## Models

All models are pre-trained and committed. No training required to run the app.

| Model | File | Purpose |
|-------|------|---------|
| LightGBM cardiovascular | `models/saved/lgbm_model_cardio.pkl` | CVD risk prediction |
| StandardScaler (cardio) | `models/saved/lgbm_scaler_cardio.pkl` | Feature scaling |
| Isolation Forest | `models/saved/iso_forest_cardio.pkl` | Anomaly detection |
| Review classifier | `backend/models/review_classifier_best.pkl` | Clean / Review / Block |
| Quality regressor | `backend/models/quality_regressor_best.pkl` | Data quality score |

To retrain from scratch:
```bash
python scripts/training_recipes/01_train_lgbm_cardio.py
```

---

## Key design decisions

### Explainability is non-negotiable
Every prediction is accompanied by a SHAP bar chart showing feature contributions. In a clinical context, a black-box score is insufficient — the clinician must be able to understand and challenge any model output.

### Saving model components separately (no sklearn Pipeline)
The LightGBM model and its StandardScaler are saved as separate files rather than a full sklearn Pipeline. Pipeline serialisation breaks across sklearn minor versions (a known "passthrough" sentinel issue in ColumnTransformer). Separating components and reconstructing preprocessing at inference time is more stable and more auditable.

### Rule-based safety overrides on top of ML
The review classifier (ML) can under-weight flags in edge cases. Two deterministic overrides are applied after every ML prediction:
1. Any validation flag present → minimum verdict is **Review** (never Clean)
2. High-severity flag or critical missing field → escalate to **Block**

This mirrors how regulatory-grade clinical data management systems work: the model informs, but hard rules protect.

### RAG instead of full PDF context
Instead of sending entire protocol PDFs to the LLM (expensive, noisy), the system builds a ChromaDB index per document and retrieves only the most relevant chunks per query. Source citations (section + page) are shown with every answer, enabling verification.

### Direct backend imports (no FastAPI dependency)
The Streamlit app imports backend modules directly rather than calling a FastAPI HTTP server. This eliminates the need to manage two processes and simplifies deployment. The FastAPI server (`backend/main.py`) is still available for API-first deployments.

---

## Project conclusions

**Data quality is the hardest problem.** Building a robust pipeline (extraction → validation → classification → safety override) turned out to be more complex than the predictive models themselves. The interaction between ML classification and rule-based overrides required careful design.

**LLM integration requires grounding.** Vanilla LLM answers hallucinate confidently. RAG with source citations and a "not found in the protocol" fallback is essential for clinical use.

**Cross-library compatibility is a real operational concern.** sklearn Pipeline serialisation, LightGBM version pinning, and sentence-transformers/torch multiprocessing on macOS all required non-trivial workarounds. Saving model components separately and pinning versions in `requirements.txt` is the right approach for reproducibility.

**Explainability and ML are complementary.** SHAP was integrated from the beginning rather than as an afterthought. Alongside the rule-based overrides, this shows that ML and interpretable systems work best together in high-stakes domains.

**Productionising research code takes significant effort.** Moving from working notebooks to a multi-user Streamlit app with caching, session state, error handling, and consistent UI required as much work as the modelling itself.

---

## Team

| Person | Contributions |
|--------|--------------|
| **Cosmina** | Cardiovascular EDA · LightGBM training · SHAP · Isolation Forest · Streamlit app · document intelligence integration · deployment |
| **Ziya** | Clinical protocol backend · ChromaDB RAG pipeline · review classifier · quality regressor · disease model registry |

---

## License

For portfolio and academic demonstration purposes.
