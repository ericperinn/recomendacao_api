import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_candidatos_para_vaga_ampla_sem_filtro(client: AsyncClient):
    """Vaga ampla sem filtro PCD deve retornar todos os candidatos."""
    resp = await client.get("/vagas/100/candidatos?filtrar_pcd=false&top_k=10")
    assert resp.status_code == 200
    data = resp.json()

    assert data["vaga_id"] == 100
    assert data["filtro_pcd_aplicado"] is False
    assert data["total_retornado"] >= 1

    ids = [c["cand_id"] for c in data["candidatos"]]
    # Candidato 1 (dev, embedding idêntico à vaga) deve estar no topo
    assert ids[0] == 1


@pytest.mark.asyncio
async def test_candidatos_para_vaga_ampla_com_filtro_pcd(client: AsyncClient):
    """Vaga ampla com filtrar_pcd=true deve retornar apenas candidatos PCD."""
    resp = await client.get("/vagas/100/candidatos?filtrar_pcd=true&top_k=10")
    assert resp.status_code == 200
    data = resp.json()

    assert data["filtro_pcd_aplicado"] is True
    for cand in data["candidatos"]:
        assert cand["is_pcd"] is True, "Com filtro PCD só candidatos PCD devem aparecer"


@pytest.mark.asyncio
async def test_candidatos_para_vaga_pcd_com_filtro(client: AsyncClient):
    """Vaga PCD exclusiva + filtrar_pcd=true → apenas candidatos PCD."""
    resp = await client.get("/vagas/200/candidatos?filtrar_pcd=true&top_k=10")
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_pcd_exclusive"] is True
    assert data["filtro_pcd_aplicado"] is True

    ids = [c["cand_id"] for c in data["candidatos"]]
    assert 2 in ids, "Candidato PCD deve aparecer para vaga PCD exclusiva"
    # Candidato 1 (não-PCD) não deve aparecer
    assert 1 not in ids


@pytest.mark.asyncio
async def test_candidatos_para_vaga_pcd_sem_filtro(client: AsyncClient):
    """Vaga PCD exclusiva + filtrar_pcd=false → base geral (recrutador desativou)."""
    resp = await client.get("/vagas/200/candidatos?filtrar_pcd=false&top_k=10")
    assert resp.status_code == 200
    data = resp.json()

    assert data["filtro_pcd_aplicado"] is False
    # Com base geral, ambos os candidatos devem aparecer
    ids = [c["cand_id"] for c in data["candidatos"]]
    assert len(ids) == 2


@pytest.mark.asyncio
async def test_vaga_nao_encontrada(client: AsyncClient):
    resp = await client.get("/vagas/99999/candidatos")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_indexar_vaga_nova(client: AsyncClient):
    """POST /vagas/indexar deve criar um novo embedding de vaga."""
    payload = {
        "vaga_id": 999,
        "titulo": "Engenheiro de Dados",
        "cidade": "Remoto",
        "estado": "BR",
        "area": "Data Engineering",
        "is_pcd_exclusive": False,
        "habilidades": ["python", "spark", "sql"],
        "conhecimentos": ["databricks", "airflow"],
        "funcao": "Engenheiro de Dados",
        "descricao": "Pipeline de dados em larga escala.",
    }
    resp = await client.post("/vagas/indexar", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["vaga_id"] == 999
    assert data["acao"] == "criado"
    assert "spark" in data["texto_gerado"]


@pytest.mark.asyncio
async def test_indexar_vaga_existente(client: AsyncClient):
    payload = {
        "vaga_id": 100,
        "titulo": "Dev Full Stack Atualizado",
        "cidade": "Manaus",
        "estado": "AM",
        "area": "Desenvolvimento",
        "is_pcd_exclusive": False,
        "habilidades": ["typescript", "react"],
        "conhecimentos": [],
        "funcao": "",
        "descricao": "",
    }
    resp = await client.post("/vagas/indexar", json=payload)
    assert resp.status_code == 201
    assert resp.json()["acao"] == "atualizado"
