FROM python:3.11-slim

WORKDIR /app

# dependências do sistema para asyncpg e sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# baixa stopwords do NLTK na build (evita download em runtime)
RUN python -c "import nltk; nltk.download('stopwords', quiet=True)"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
