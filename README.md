# API de Recomendação — TCC

API FastAPI para recomendação semântica de vagas e candidatos usando SBERT.

## Estrutura

```
recomendacao_api/
├── app/
│   ├── main.py                     # FastAPI app
│   ├── core/
│   │   ├── config.py               # Settings via .env
│   │   └── database.py             # SQLAlchemy async
│   ├── models/
│   │   └── schema.py               # ORM (espelho dos CSVs + tabelas de embedding)
│   ├── services/
│   │   ├── text_builder.py         # monta texto candidato/vaga (lógica do notebook)
│   │   ├── embedder.py             # singleton SBERT, trocável via .env
│   │   └── recommender.py          # cosine similarity + regras PCD
│   ├── routers/
│   │   ├── candidatos.py           # GET /candidatos/{id}/vagas
│   │   ├── vagas.py                # GET /vagas/{id}/candidatos
│   │   └── admin.py                # POST /admin/recalcular-embeddings
│   └── scripts/
│       └── seed_and_embed.py       # importa CSVs + gera embeddings iniciais
├── dados/                          # coloque os CSVs aqui
├── modelos/                        # coloque o modelo fine-tuned aqui (opcional)
├── docker-compose.yml
├── Dockerfile
└── .env.example
```

## Setup rápido (Docker)

```bash
# 1. Clone / copie o projeto
cp .env.example .env

# 2. Coloque os CSVs em ./dados/
#    Os arquivos esperados são os mesmos do notebook:
#    candidatos.csv, candidato_experiencias.csv, candidato_formacoes.csv,
#    candidato_deficiencias.csv, deficiencias.csv,
#    candidato_habilidades_competencias.csv, candidato_adaptacoes.csv,
#    vagas.csv, vaga_habilidades_competencias.csv, vaga_conhecimentos_tecnicos.csv,
#    vaga_funcoes.csv, areas_atuacao.csv

# 3. (Opcional) coloque o modelo fine-tuned em ./modelos/sbert_tcc_recrutamento_pt/
#    e atualize MODEL_PATH no .env

# 4. Sobe o banco e a API
docker compose up -d db_dados db_embeddings rabbitmq
docker compose up -d api worker

# 5. Executa o seed (importa CSVs + gera embeddings — leva alguns minutos pelo SBERT)
docker compose run --rm api python -m app.scripts.seed_and_embed
```

## Endpoints

### `GET /candidatos/{cand_id}/vagas`
Retorna vagas recomendadas para um candidato.

```
GET /candidatos/1/vagas?top_k=10
```

**Resposta:**
```json
{
  "cand_id": 1,
  "is_pcd": false,
  "total_retornado": 5,
  "vagas": [
    {
      "vaga_id": 91001,
      "titulo": "Dev Full Stack Python/TS",
      "area": "Desenvolvimento",
      "is_pcd_exclusive": false,
      "score": 0.5235
    }
  ]
}
```

### `GET /vagas/{vaga_id}/candidatos`
Retorna candidatos ranqueados para uma vaga (visão do recrutador).

```
GET /vagas/91042/candidatos?top_k=10&filtrar_pcd=true
```

**Resposta:**
```json
{
  "vaga_id": 91042,
  "vaga_titulo": "Vaga PCD - Motorista (CNH D)",
  "is_pcd_exclusive": true,
  "filtro_pcd_aplicado": true,
  "total_retornado": 3,
  "candidatos": [
    {
      "cand_id": 900009,
      "is_pcd": true,
      "score": 0.6741
    }
  ]
}
```

### `POST /admin/recalcular-embeddings`
Regenera todos os embeddings. Use após trocar o modelo ou inserir novos dados em lote.

```bash
curl -X POST http://localhost:8000/admin/recalcular-embeddings \
  -H "x-admin-token: troque_em_producao" \
  -H "Content-Type: application/json" \
  -d '{"entidade": "ambos"}'
```

### `GET /admin/status`
Mostra qual modelo está carregado e quantos embeddings existem.

```bash
curl http://localhost:8000/admin/status \
  -H "x-admin-token: troque_em_producao"

### `GET /admin/infra-status`
Mostra a saúde operacional dos dois bancos, broker RabbitMQ e backlog da fila.

```bash
curl http://localhost:8000/admin/infra-status \
  -H "x-admin-token: troque_em_producao"
```

### `POST /candidatos/indexar` e `POST /vagas/indexar`
Agora esses endpoints **enfileiram** indexação assíncrona (HTTP 202) em vez de recalcular online.
O processamento ocorre no serviço `worker` via RabbitMQ.
```

## Trocando o modelo

```bash
# 1. Atualize MODEL_PATH no .env
#    Para fine-tuned: MODEL_PATH=./modelos/sbert_tcc_recrutamento_pt
#    Para base:       MODEL_PATH=paraphrase-multilingual-mpnet-base-v2

# 2. Regenere os embeddings
curl -X POST http://localhost:8000/admin/recalcular-embeddings \
  -H "x-admin-token: troque_em_producao" \
  -d '{"entidade": "ambos"}'
```

## Quando o schema oficial chegar

1. Atualize os nomes de coluna em `app/models/schema.py` (tabelas ORM)
2. Atualize os nomes de coluna em `app/services/text_builder.py` (montagem do texto)
3. Reexecute o seed: `python -m app.scripts.seed_and_embed`
4. A lógica de recomendação não muda — só o mapeamento de dados.

## Documentação interativa

Com a API rodando: http://localhost:8000/docs
