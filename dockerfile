# ---- Basis ----
FROM python:3.11-slim

# ---- System-Libs für OpenCV + PDF/OCR ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    poppler-utils            \
    tesseract-ocr            \
    ghostscript              \
    && rm -rf /var/lib/apt/lists/*

# ---- Python-Layer ----
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# ---- Streamlit ----
ENV PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
