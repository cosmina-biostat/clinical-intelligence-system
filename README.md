# Clinical Data Intelligence & Monitoring System
> Capstone project — Healthcare AI | 3-person team

A unified platform that combines **disease risk prediction**, **anomaly detection**, and **clinical document Q&A** using classical ML, LLMs, and RAG.

---

## What it does
| Module | Description |
|--------|-------------|
| **Risk prediction** | XGBoost / LightGBM / RF / LogReg on 3 clinical datasets |
| **SHAP explanations** | Plain-English feature importance for every prediction |
| **Anomaly detection** | Isolation Forest flags unusual patient profiles |
| **RAG Q&A** | Upload a clinical PDF, ask questions, Claude answers |
| **Model comparison** | Side-by-side AUC / F1 across all classifiers and datasets |

## Datasets
- [Heart Disease UCI](https://www.kaggle.com/datasets/ronitf/heart-disease-uci) — cardiovascular risk
- [MS CIS Conversion](https://www.kaggle.com/) — multiple sclerosis progression
- [ISIC Melanoma 2020](https://www.kaggle.com/c/siim-isic-melanoma-classification) — skin lesion classification
- Synthetic 300+ patients (generated via `src/data/synthetic.py`)

## Tech stack
```
Python 3.11        scikit-learn    XGBoost     LightGBM
SHAP               MLflow          FAISS       LangChain
sentence-transformers  pdfplumber  Streamlit   Anthropic API
```

## Quickstart
```bash
git clone https://github.com/YOUR_USERNAME/clinical-intelligence-system.git
cd clinical-intelligence-system
pip install -r requirements.txt
cp .env.example .env          # add your API keys
python -m src.data.synthetic  # generate synthetic data
# Run notebooks 01–07 in order
streamlit run app/app.py
```

See [docs/setup.md](docs/setup.md) for the full guide.

## Team
| Role | Responsibilities |
|------|-----------------|
| Person A — ML engineer | Data prep, classification models, SHAP, MLflow |
| Person B — AI engineer | PDF parsing, RAG pipeline, anomaly detection |
| Person C — Full-stack  | Streamlit dashboard, Docker, testing, docs |

## Project structure
```
clinical-intelligence-system/
├── app/                  # Streamlit dashboard
│   └── pages/            # One file per tab
├── src/
│   ├── data/             # Loaders, preprocessor, synthetic generator
│   ├── models/           # Train, evaluate, explain
│   ├── rag/              # PDF parser, embedder, retriever
│   ├── anomaly/          # Isolation Forest detector
│   └── utils/            # Config, logger
├── notebooks/            # 01 EDA → 07 model comparison
├── models/saved/         # .pkl files (gitignored — share via Drive)
├── data/                 # raw / processed / synthetic (gitignored)
├── reports/              # CSV results + figures
├── tests/                # pytest test suite
└── docs/                 # Setup, architecture, team workflow
```

## Results (to be filled after training)
| Dataset | Best model | AUC | F1 |
|---------|-----------|-----|----|
| Cardiovascular | — | — | — |
| MS / CIS | — | — | — |
| Melanoma | — | — | — |
