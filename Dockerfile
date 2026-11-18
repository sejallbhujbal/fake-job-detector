# Works locally and as a Hugging Face "Docker" Space (port 7860, user 1000).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/local/share/nltk_data \
    PORT=7860 \
    APP_MODE=ui

# libgomp1 is required by XGBoost on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements-app.txt .
RUN pip install -r requirements-app.txt \
    && python -c "import nltk; [nltk.download(p, download_dir='/usr/local/share/nltk_data', quiet=True) for p in ('wordnet', 'omw-1.4')]"

COPY --chown=user . .
RUN chmod +x start.sh
USER user

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request,os; urllib.request.urlopen('http://localhost:'+os.environ.get('PORT','7860')+('/health' if os.environ.get('APP_MODE')=='api' else '/_stcore/health'))" || exit 1
CMD ["bash", "start.sh"]
