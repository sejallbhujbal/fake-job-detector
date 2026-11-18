#!/usr/bin/env bash
# Container entry point.
#   APP_MODE=ui   -> Streamlit on $PORT, model runs in-process (default, HF Spaces)
#   APP_MODE=api  -> FastAPI on $PORT (for the Chrome extension)
#   APP_MODE=both -> FastAPI on 8000 + Streamlit on $PORT calling the API
set -euo pipefail
PORT="${PORT:-7860}"
MODE="${APP_MODE:-ui}"

if [ ! -f models/best_classical.joblib ]; then
  echo "WARNING: models/best_classical.joblib not found -> app runs in RULES-ONLY mode."
  echo "         Train locally (python -m src.train_classical) and include models/ in the image."
fi

case "$MODE" in
  api)
    exec uvicorn api.main:app --host 0.0.0.0 --port "$PORT"
    ;;
  both)
    uvicorn api.main:app --host 0.0.0.0 --port 8000 &
    export API_URL="http://localhost:8000"
    exec streamlit run ui/streamlit_app.py --server.port "$PORT" --server.address 0.0.0.0 --server.headless true
    ;;
  *)
    exec streamlit run ui/streamlit_app.py --server.port "$PORT" --server.address 0.0.0.0 --server.headless true
    ;;
esac
