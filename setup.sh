#!/bin/bash

set -e

echo "Setting up Docker"

docker compose up -d


echo "📦 Checking Python dependencies..."

# install only if missing (faster, safer)
pip install -r requirements.txt

echo "📦 Ensuring unstructured PDF support..."
pip install "unstructured[pdf]" --quiet

echo "🧱 Checking system dependencies..."

if ! dpkg -s libgl1 libglib2.0-0 tesseract-ocr >/dev/null 2>&1; then
    echo "Installing system dependencies..."
    sudo apt update
    sudo apt install -y libgl1 libglib2.0-0 tesseract-ocr
else
    echo "System dependencies already installed ✔"
fi

echo "🚀 Running ingestion..."
python ingest.py

echo "🎯 Starting Streamlit UI..."
python -m streamlit run ui.py --server.address=0.0.0.0 --server.port=8501
