.PHONY: install setup data train anomaly rag app test clean

install:
	pip install -r requirements.txt

setup:
	mkdir -p data/raw data/processed data/synthetic models/saved reports/figures
	cp .env.example .env
	@echo "Edit .env with your API keys before proceeding"

data:
	python -m src.data.synthetic

train:
	jupyter nbconvert --to notebook --execute notebooks/04_train_models.ipynb

anomaly:
	jupyter nbconvert --to notebook --execute notebooks/05_anomaly_detection.ipynb

rag:
	jupyter nbconvert --to notebook --execute notebooks/06_rag_pipeline.ipynb

app:
	streamlit run app/app.py

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
