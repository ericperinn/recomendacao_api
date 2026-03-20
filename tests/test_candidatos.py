import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_vagas_recomendadas_candidato_normal(client: AsyncClient):
    """Candidato não-PCD deve receber vagas amplas, não ver vagas exclusivas PCD."""
    resp = await client.get("/candidatos/1/vagas?top_k=10")
    assert resp.status_code == 200
    data = resp.json()

    assert data["cand_id"] == 1
    assert data["is_pcd"] is False
    assert data["total_retornado"] >= 1

    # Nenhuma vaga exclusiva PCD deve aparecer
    pcd_exclusivas = [v for v in data["vagas"] if v["is_pcd_exclusive"]]
    assert pcd_exclusivas == [], "Candidato não-PCD não deve ver vagas PCD exclusivas"

    # A vaga de dev deve estar no topo (mesmo embedding → score máximo)
    assert data["vagas"][0]["vaga_id"] == 100


@pytest.mark.asyncio
async def test_vagas_recomendadas_candidato_pcd(client: AsyncClient):
    """Candidato PCD deve ver vagas exclusivas PCD no topo (bônus +0.25)."""
    resp = await client.get("/candidatos/2/vagas?top_k=10")
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_pcd"] is True
    assert data["total_retornado"] >= 1

    # A vaga PCD deve aparecer (e com bônus deve estar no topo)
    ids = [v["vaga_id"] for v in data["vagas"]]
    assert 200 in ids, "Vaga PCD exclusiva deve aparecer para candidato PCD"


@pytest.mark.asyncio
async def test_candidato_nao_encontrado(client: AsyncClient):
    resp = await client.get("/candidatos/99999/vagas")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_top_k_limita_resultados(client: AsyncClient):
    resp = await client.get("/candidatos/1/vagas?top_k=1")
    assert resp.status_code == 200
    assert len(resp.json()["vagas"]) <= 1


@pytest.mark.asyncio
async def test_indexar_candidato_novo(client: AsyncClient):
    """POST /candidatos/indexar deve criar um novo embedding."""
    payload = {
        "cand_id": 999,
        "nome": "Teste Novo",
        "cidade": "Manaus",
        "estado": "AM",
        "is_pcd": False,
        "habilidades": ["python", "fastapi", "sql"],
        "experiencias": ["Desenvolvedor Backend"],
        "formacoes": ["Engenharia de Computação"],
        "deficiencias": [],
    }
    resp = await client.post("/candidatos/indexar", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["cand_id"] == 999
    assert data["acao"] == "criado"
    assert "python" in data["texto_gerado"]


@pytest.mark.asyncio
async def test_indexar_candidato_existente(client: AsyncClient):
    """POST /candidatos/indexar em ID existente deve atualizar (não duplicar)."""
    payload = {
        "cand_id": 1,
        "nome": "Candidato Atualizado",
        "cidade": "Manaus",
        "estado": "AM",
        "is_pcd": False,
        "habilidades": ["java", "spring"],
        "experiencias": [],
        "formacoes": [],
        "deficiencias": [],
    }
    resp = await client.post("/candidatos/indexar", json=payload)
    assert resp.status_code == 201
    assert resp.json()["acao"] == "atualizado"
