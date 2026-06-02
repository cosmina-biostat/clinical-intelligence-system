# Setup guide

## 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/clinical-intelligence-system.git
cd clinical-intelligence-system
```

## 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

## 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 4. Configure API keys
```bash
cp .env.example .env
# Open .env and paste your Anthropic API key and Kaggle credentials
```

## 5. Download datasets
Go to Kaggle and download these three datasets manually into `data/raw/`:
- Heart Disease UCI → save as `heart_disease.csv`
- MS CIS Conversion → save as `ms_cis.csv`
- ISIC Melanoma 2020 → save as `melanoma.csv`

## 6. Generate synthetic data
```bash
python -m src.data.synthetic
```

## 7. Run the notebooks in order
Open Jupyter and run notebooks 01 through 07 in sequence.

## 8. Launch the dashboard
```bash
streamlit run app/app.py
```
