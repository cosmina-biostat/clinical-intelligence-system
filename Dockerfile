FROM python:3.11-slim

WORKDIR /app

# System dependencies for pdfplumber / PyMuPDF / faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit config — disable telemetry & set default port
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

# ANTHROPIC_API_KEY must be passed at runtime:
#   docker run -e ANTHROPIC_API_KEY=sk-... -p 8501:8501 clinical-intelligence
CMD ["streamlit", "run", "app/app.py"]
