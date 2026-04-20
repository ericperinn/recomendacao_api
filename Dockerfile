FROM python:3.11-slim

WORKDIR /app

ENV PIP_DEFAULT_TIMEOUT=1200 \
    PIP_RETRIES=20 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
COPY constraints.txt .

RUN pip install --no-cache-dir --retries 20 --timeout 1200 \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

RUN python -c "import nltk; nltk.download('stopwords', quiet=True)"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", 