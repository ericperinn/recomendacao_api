"""
Script de diagnóstico — roda o commit tabela por tabela para identificar
qual tabela/coluna causa o DataError do asyncpg.
"""

import asyncio
import json
from pathlib import Path
from datetime import date as pydate

import pandas as pd
from sqlalchemy import text
from sqlalchemy.types import String, Boolean, Integer, Numeric, Date as SADate

from app.core.database import engine, AsyncSessionLocal, Base
from app.models.schema import (
    Candidato,
    Empresa,
    CandidatoHabilidade,
    CandidatoOutraHabilidade,
    CandidatoExperiencia,
    CandidatoFormacao,
    CandidatoDeficiencia,
    CandidatoOutraDeficiencia,
    CandidatoAdaptacao,
    CandidatoIdioma,
    Idioma,
    Deficiencia,
    NivelEscolaridade,
    AreaAtuacao,
    VagaFuncao,
    VagaModalidade,
    VagaRegime,
    TurnoTrabalho,
    Genero,
    Vaga,
    VagaHabilidade,
    VagaConhecimento,
    VagaDiferencial,
    VagaCertificacao,
    VagaOutraAreaAtuacao,
    VagaOutroRegime,
    CandidatoEmbedding,
    VagaEmbedding,
)

CSV_DIR = Path("./dados")

ORDER = [
    ("generos", Genero),
    ("idiomas", Idioma),
    ("deficiencias", Deficiencia),
    ("niveis_escolaridade", NivelEscolaridade),
    ("areas_atuacao", AreaAtuacao),
    ("vaga_funcoes", VagaFuncao),
    ("vaga_modalidades", VagaModalidade),
    ("vaga_regimes", VagaRegime),
    ("turnos_trabalho", TurnoTrabalho),
    ("empresas", Empresa),
    ("candidatos", Candidato),
    ("candidato_habilidades_competencias", CandidatoHabilidade),
    ("candidato_outras_habilidades", CandidatoOutraHabilidade),
    ("candidato_experiencias", CandidatoExperiencia),
    ("candidato_formacoes", CandidatoFormacao),
    ("candidato_deficiencias", CandidatoDeficiencia),
    ("candidato_outras_deficiencias", CandidatoOutraDeficiencia),
    ("candidato_adaptacoes", CandidatoAdaptacao),
    ("candidato_idiomas", CandidatoIdioma),
    ("vagas", Vaga),
    ("vaga_habilidades_competencias", VagaHabilidade),
    ("vaga_conhecimentos_tecnicos", VagaConhecimento),
    ("vaga_diferenciais", VagaDiferencial),
    ("vaga_certificacoes", VagaCertificacao),
    ("vaga_outra_area_atuacao", VagaOutraAreaAtuacao),
    ("vaga_outro_regime", VagaOutroRegime),
]

DTYPE_MAP = {
    "empresas": {"empr_cnpj": str, "empr_cep": str, "empr_numero": str},
    "candidatos": {"cand_cpf": str, "cand_cep": str, "cand_numero": str},
}


def build_record(row, modelo_orm):
    orm_cols = {c.name: type(c.type) for c in modelo_orm.__table__.columns}
    clean = {}
    for k, v in row.items():
        if k not in orm_cols:
            continue
        if v is None or v == "":
            clean[k] = None
            continue
        ctype = orm_cols[k]
        if issubclass(ctype, Boolean):
            clean[k] = (
                v.lower() in ("true", "1", "t", "sim", "s")
                if isinstance(v, str)
                else bool(v)
            )
        elif issubclass(ctype, SADate):
            if isinstance(v, str) and v:
                try:
                    clean[k] = pydate.fromisoformat(v.split("T")[0])
                except ValueError:
                    clean[k] = None
            else:
                clean[k] = None
        elif issubclass(ctype, Integer):
            clean[k] = int(float(v))
        elif issubclass(ctype, Numeric):
            clean[k] = float(v)
        else:
            clean[k] = (
                str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
            )
    return modelo_orm(**clean)


async def main():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    all_tables = " ,".join(m.__tablename__ for _, m in ORDER)
    extra = "candidato_embeddings, vaga_embeddings"
    async with AsyncSessionLocal() as db:
        await db.execute(
            text(f"TRUNCATE TABLE {extra}, {all_tables} RESTART IDENTITY CASCADE")
        )
        await db.commit()
    print("Truncated OK")

    for nome, modelo in ORDER:
        path = CSV_DIR / f"{nome}.csv"
        if not path.exists():
            print(f"SKIP {nome} (not found)")
            continue
        df = pd.read_csv(path, keep_default_na=False)
        if nome in DTYPE_MAP:
            for col, _ in DTYPE_MAP[nome].items():
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(r"\.0$", "", regex=True)
        if df.empty:
            print(f"SKIP {nome} (empty)")
            continue

        records_json = json.loads(df.to_json(orient="records"))

        async with AsyncSessionLocal() as db:
            regs = []
            for i, r in enumerate(records_json):
                try:
                    regs.append(build_record(r, modelo))
                except Exception as e:
                    print(f"  BUILD ERROR {nome}[{i}]: {e}")
            db.add_all(regs)
            try:
                await db.commit()
                print(f"OK {nome}: {len(regs)} rows")
            except Exception as e:
                await db.rollback()
                print(f"COMMIT ERROR {nome}: {e}")
                # Try row by row to find culprit
                async with AsyncSessionLocal() as db2:
                    for i, r in enumerate(records_json):
                        try:
                            obj = build_record(r, modelo)
                            db2.add(obj)
                            await db2.flush()
                        except Exception as e2:
                            print(f"  ROW ERROR {nome}[{i}]: {e2}")
                            print(f"  ROW DATA: {dict(list(r.items())[:8])}")
                            await db2.rollback()
                    try:
                        await db2.commit()
                    except:
                        pass


if __name__ == "__main__":
    asyncio.run(main())
