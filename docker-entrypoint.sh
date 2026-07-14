#!/bin/sh
set -e

MODE="${1:-api}"

if [ "$MODE" = "api" ]; then
  exec uvicorn app.api:app --host 0.0.0.0 --port 8000
elif [ "$MODE" = "ui" ]; then
  exec streamlit run frontend/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
elif [ "$MODE" = "ingest" ]; then
  exec python -c "from app.rag_pipeline import RAGPipeline; print(RAGPipeline(auto_load=False).ingest_directory())"
else
  exec "$@"
fi
