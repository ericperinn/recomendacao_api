"""
Script de seed — importa os CSVs para o banco e calcula embeddings iniciais.

Uso:
  docker compose run --rm api python -m app.scripts.seed_and_embed

Os CSVs devem estar em ./dados/ com os nomes exatos das tabelas do schema Prisma.
"""

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import engine, AsyncSessionLocal, Base
from app.models.schema import (
    Candidato, Empresa, CandidatoHabilidade, CandidatoOutraHabilidade,
    CandidatoExperiencia, CandidatoFormacao, CandidatoDeficiencia,
    CandidatoOutraDeficiencia, CandidatoAdaptacao, CandidatoIdioma,
    Idioma, Deficiencia, NivelEscolaridade, AreaAtuacao, VagaFuncao,
    VagaModalidade, VagaRegime, TurnoTrabalho, Genero, Vaga,
    VagaHabilidade, VagaConhecimento, VagaDiferencial, VagaCertificacao,
    VagaOutraAreaAtuacao, VagaOutroRegime, CandidatoEmbedding, VagaEmbedding,
)
from app.services.text_builder import (
    montar_texto_candidato, montar_texto_vaga, is_candidato_pcd, is_vaga_pcd, limpar_texto,
)
from app.services.embedder import EmbedderService

CSV_DIR = Path("./dados")

CSV_LOAD_ORDER = [
    "generos", "idiomas", "deficiencias", "niveis_escolaridade", "areas_atuacao",
    "vaga_funcoes", "vaga_modalidades", "vaga_regimes", "turnos_trabalho",
    "empresas", "candidatos", "candidato_habilidades_competencias",
    "candidato_outras_habilidades", "candidato_experiencias", "candidato_formacoes",
    "candidato_deficiencias", "candidato_outras_deficiencias", "candidato_adaptacoes",
    "candidato_idiomas", "vagas", "vaga_habilidades_competencias",
    "vaga_conhecimentos_tecnicos", "vaga_diferenciais", "vaga_certificacoes",
    "vaga_outra_area_atuacao", "vaga_outro_regime",
]

ORM_MAP = {
    "generos": Genero, "idiomas": Idioma, "deficiencias": Deficiencia,
    "niveis_escolaridade": NivelEscolaridade, "areas_atuacao": AreaAtuacao,
    "vaga_funcoes": VagaFuncao, "vaga_modalidades": VagaModalidade,
    "vaga_regimes": VagaRegime, "turnos_trabalho": TurnoTrabalho,
    "empresas": Empresa, "candidatos": Candidato,
    "candidato_habilidades_competencias": CandidatoHabilidade,
    "candidato_outras_habilidades": CandidatoOutraHabilidade,
    "candidato_experiencias": CandidatoExperiencia,
    "candidato_formacoes": CandidatoFormacao,
    "candidato_deficiencias": CandidatoDeficiencia,
    "candidato_outras_deficiencias": CandidatoOutraDeficiencia,
    "candidato_adaptacoes": CandidatoAdaptacao, "candidato_idiomas": CandidatoIdioma,
    "vagas": Vaga, "vaga_habilidades_competencias": VagaHabilidade,
    "vaga_conhecimentos_tecnicos": VagaConhecimento, "vaga_diferenciais": VagaDiferencial,
    "vaga_certificacoes": VagaCertificacao, "vaga_outra_area_atuacao": VagaOutraAreaAtuacao,
    "vaga_outro_regime": VagaOutroRegime,
}

DTYPE_MAPPING = {
    "empresas": {"empr_cnpj": str, "empr_cep": str, "empr_numero": str},
    "candidatos": {"cand_cpf": str, "cand_cep": str, "cand_numero": str},
}

def _load_csvs() -> dict[str, pd.DataFrame]:
    dfs = {}
    for name in CSV_LOAD_ORDER:
        path = CSV_DIR / f"{name}.csv"
        if path.exists():
            dfs[name] = pd.read_csv(path, keep_default_na=False)
            if name in DTYPE_MAPPING:
                for col in DTYPE_MAPPING[name].keys():
                    if col in dfs[name].columns:
                        dfs[name][col] = dfs[name][col].astype(str).str.replace(r"\.0$", "", regex=True)
            print(f"  ✅ {name}.csv → {dfs[name].shape}")
        else:
            dfs[name] = pd.DataFrame()
    return dfs

async def _inserir_csv(db: AsyncSession, nome: str, df: pd.DataFrame, modelo_orm):
    if df.empty:
        return

    # REGRA 1: Se for a tabela candidatos, remove e-mails duplicados mantendo o primeiro
    if nome == "candidatos" and "cand_email" in df.columns:
        df = df.drop_duplicates(subset=["cand_email"], keep="first")

    df = df.replace({np.nan: None})
    records = df.to_dict(orient="records")
    registros = []

    from sqlalchemy.types import Date, DateTime, Boolean, Integer, Numeric, String
    # Pega o tipo exato da coluna no SQLAlchemy
    cols = {c.name: type(c.type) for c in modelo_orm.__table__.columns}

    erros = 0
    for idx, row in enumerate(records):
        try:
            clean_kwargs = {}
            for k, v in row.items():
                if k not in cols:
                    continue

                ctype = cols[k]

                # REGRA 2: Tratar nulos e strings vazias
                if pd.isna(v) or v is None or str(v).strip() == "" or str(v).lower() in ("nan", "none", "null"):
                    # Se o banco espera String, manda vazio em vez de NULL para evitar NotNullViolation
                    if issubclass(ctype, String):
                        clean_kwargs[k] = "" 
                    else:
                        clean_kwargs[k] = None
                    continue

                try:
                    if issubclass(ctype, (Date, DateTime)):
                        dt = pd.to_datetime(str(v), errors='coerce')
                        if pd.isna(dt):
                            clean_kwargs[k] = None
                        else:
                            clean_kwargs[k] = dt.to_pydatetime() if issubclass(ctype, DateTime) else dt.date()
                    elif issubclass(ctype, Boolean):
                        clean_kwargs[k] = str(v).lower() in ("true", "1", "t", "sim", "s")
                    elif issubclass(ctype, Integer):
                        clean_kwargs[k] = int(float(v))
                    elif issubclass(ctype, Numeric):
                        clean_kwargs[k] = float(v)
                    else:
                        if isinstance(v, float) and v.is_integer():
                            clean_kwargs[k] = str(int(v))
                        else:
                            clean_kwargs[k] = str(v)
                except Exception:
                    clean_kwargs[k] = None

            registros.append(modelo_orm(**clean_kwargs))
        except Exception as e:
            erros += 1
            if erros <= 3:
                print(f"  ⚠️  {nome} linha {idx}: {e}")

    if erros > 3:
        print(f"  ⚠️  {nome}: {erros} linhas ignoradas por erro de parsing.")

    db.add_all(registros)
    print(f"  → {len(registros)} registros carregados para a tabela {modelo_orm.__tablename__}")

async def _truncar_tabelas(db: AsyncSession):
    tabelas = " ,".join(ORM_MAP[nome].__tablename__ for nome in reversed(CSV_LOAD_ORDER) if nome in ORM_MAP)
    tabelas_embedding = "candidato_embeddings, vaga_embeddings"
    await db.execute(text(f"TRUNCATE TABLE {tabelas_embedding}, {tabelas} RESTART IDENTITY CASCADE"))
    await db.commit()
    print("  ✅ Tabelas truncadas com RESTART IDENTITY CASCADE.")

async def embed_candidatos(db: AsyncSession, embedder: EmbedderService) -> int:
    from sqlalchemy import select
    cands = (await db.execute(select(Candidato))).scalars().all()
    cand_ids = [c.cand_id_candidato for c in cands]
    if not cand_ids: return 0

    dfs = _load_csvs()
    textos, ids_validos, pcd_flags = [], [], []
    for cid in cand_ids:
        texto = montar_texto_candidato(cid, dfs)
        if texto:
            textos.append(limpar_texto(texto))
            ids_validos.append(cid)
            pcd_flags.append(is_candidato_pcd(cid, dfs))

    if not textos: return 0

    print(f"  Encodando {len(textos)} embeddings de candidatos...")
    embeddings = embedder.encode(textos)
    await db.execute(delete(CandidatoEmbedding))
    for cid, texto, emb, pcd in zip(ids_validos, textos, embeddings, pcd_flags):
        db.add(CandidatoEmbedding(cand_id=cid, texto_limpo=texto, is_pcd=bool(pcd), embedding=emb.astype(np.float32).tobytes(), modelo_usado=embedder.model_path))
    await db.commit()
    return len(ids_validos)

async def embed_vagas(db: AsyncSession, embedder: EmbedderService) -> int:
    from sqlalchemy import select
    vagas = (await db.execute(select(Vaga))).scalars().all()
    if not vagas: return 0

    dfs = _load_csvs()
    textos, metas = [], []
    for vaga in vagas:
        texto, area = montar_texto_vaga(vaga.vaga_id_vaga, vaga.vaga_id_empresa, dfs)
        if texto:
            textos.append(limpar_texto(texto))
            metas.append({"vaga_id": vaga.vaga_id_vaga, "vaga_id_empresa": vaga.vaga_id_empresa, "titulo": vaga.vaga_titulo, "area": area or "", "is_pcd": is_vaga_pcd(vaga.vaga_titulo)})

    if not textos: return 0

    print(f"  Encodando {len(textos)} embeddings de vagas...")
    embeddings = embedder.encode(textos)
    await db.execute(delete(VagaEmbedding))
    for texto, emb, meta in zip(textos, embeddings, metas):
        db.add(VagaEmbedding(vaga_id=meta["vaga_id"], vaga_id_empresa=meta["vaga_id_empresa"], vaga_titulo=meta["titulo"], vaga_area=meta["area"], texto_limpo=texto, is_pcd_exclusive=meta["is_pcd"], embedding=emb.astype(np.float32).tobytes(), modelo_usado=embedder.model_path))
    await db.commit()
    return len(metas)

async def main():
    print("=" * 55)
    print("SEED — Importando CSVs e gerando embeddings")
    print("=" * 55)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as db:
        print("[1/3] Limpando dados existentes...")
        await _truncar_tabelas(db)

        print("\n[2/3] Importando CSVs...")
        dfs = _load_csvs()
        
        # Inserimos as tabelas em ordem
        for nome in CSV_LOAD_ORDER:
            if nome in ORM_MAP and nome in dfs:
                
                # PROTEÇÃO ANTI-ÓRFÃOS
                # Antes de inserir tabelas filhas de candidatos, garantimos que o ID do candidato existe no banco
                if nome in ["candidato_experiencias", "candidato_formacoes", "candidato_deficiencias", "candidato_adaptacoes", "candidato_idiomas"]:
                    from sqlalchemy import select
                    from app.models.schema import Candidato
                    
                    # Busca todos os IDs de candidatos que foram inseridos com sucesso
                    result = await db.execute(select(Candidato.cand_id_candidato))
                    ids_validos = [row[0] for row in result.all()]
                    
                    # Identifica qual é a coluna de FK no DataFrame atual
                    fk_col = next((col for col in dfs[nome].columns if "id_candidato" in col), None)
                    
                    if fk_col and not dfs[nome].empty:
                        tamanho_original = len(dfs[nome])
                        # Filtra o DataFrame, mantendo apenas linhas onde a FK está na lista de IDs válidos
                        dfs[nome] = dfs[nome][dfs[nome][fk_col].isin(ids_validos)]
                        tamanho_novo = len(dfs[nome])
                        
                        if tamanho_original != tamanho_novo:
                            print(f"  🧹 {nome}: Removidos {tamanho_original - tamanho_novo} registros órfãos (candidato não existe).")


                await _inserir_csv(db, nome, dfs[nome], ORM_MAP[nome])
                try:
                    await db.commit() 
                except Exception as e:
                    print(f"  ❌ ERRO GRAVE ao salvar a tabela {nome}: {e}")
                    await db.rollback()

        print("\n[3/3] Carregando modelo SBERT...")
        embedder = EmbedderService.get()
        n_cand = await embed_candidatos(db, embedder)
        n_vaga = await embed_vagas(db, embedder)

    print("\n" + "=" * 55)
    print(f"✅ Seed concluído: {n_cand} candidatos, {n_vaga} vagas.")
    print("=" * 55)

if __name__ == "__main__":
    asyncio.run(main())