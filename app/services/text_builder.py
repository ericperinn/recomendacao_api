"""
Monta o texto semântico de candidatos e vagas a partir das tabelas relacionais.
Nomes de coluna alinhados com o schema Prisma oficial do sistema original.

Ordem de prioridade no texto (maior peso semântico primeiro):
  Candidato: habilidades → outras_habilidades → experiências → formações → deficiências → adaptações → local
    Vaga:      título → habilidades → conhecimentos → diferenciais → certificações → função/área → escolaridade/modalidade/regime → descrição → local
"""
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

_STOPWORDS_PT = set(stopwords.words("portuguese"))
_STOPWORDS_CUSTOM = {
    # Localização — filtro rígido feito via banco, não precisa estar no texto
    "manaus", "am", "amazonas",
    # Rótulos estruturais que não carregam semântica de cargo
    "título", "descrição", "função", "área", "atuação",
    "escolaridade", "mínima", "desejada",
    "vaga", "vagas", "oportunidade", "oportunidades",
    "exclusiva", "exclusivo", "pcd",
    # Soft skills genéricas — foco em hard skills para o SBERT
    "comunicação", "proatividade", "organização", "dinamismo",
    "resolução", "problemas", "desejável",
    "benefícios", "salário", "crescimento", "carreira",
    # Verbos de ligação sem semântica de cargo
    "realizar", "atuar", "apoiar", "participar",
    "conhecimento", "experiência", "trabalhar", "processo",
}
STOPWORDS_FINAL = _STOPWORDS_PT | _STOPWORDS_CUSTOM


def limpar_texto(texto: str) -> str:
    texto = str(texto).lower()
    texto = re.sub(r"[^a-z0-9áéíóúâêîôûàãõç\s\-]", "", texto)
    tokens = [w for w in texto.split() if w not in STOPWORDS_FINAL and len(w) > 2]
    return " ".join(tokens)


def _expandir_descricao_vaga_curta(
    titulo: str,
    funcao: str,
    area: str | None,
    modalidade: str | None,
    escolaridade_desejada: str | None,
    regime: str | None,
    habilidades: list[str],
    conhecimentos: list[str],
    diferenciais: list[str],
    certificacoes: list[str],
    descricao: str | None,
) -> str:
    descricao_base = str(descricao or "").strip()
    if len(descricao_base) >= 180:
        return descricao_base

    termos = [
        *[str(item).strip() for item in habilidades[:4] if str(item).strip()],
        *[str(item).strip() for item in conhecimentos[:3] if str(item).strip()],
        *[str(item).strip() for item in diferenciais[:2] if str(item).strip()],
        *[str(item).strip() for item in certificacoes[:2] if str(item).strip()],
    ]

    contexto = " ".join(
        part
        for part in [
            f"funcao {funcao or titulo}" if (funcao or titulo) else "",
            f"area {area}" if area else "",
            f"modalidade {modalidade}" if modalidade else "",
            f"escolaridade desejada {escolaridade_desejada}" if escolaridade_desejada else "",
            f"regime {regime}" if regime else "",
            f"requer {', '.join(termos)}" if termos else "",
        ]
        if part
    )

    bloco_extra = (
        "Atividades e contexto: a pessoa contratada atuara no escopo da vaga, "
        "executando rotinas relacionadas ao cargo, colaborando com o time e mantendo foco em qualidade."
    )
    if contexto:
        bloco_extra = f"{bloco_extra} {contexto}."

    return f"{descricao_base} {bloco_extra}".strip()[:800]


# ── Candidato ────────────────────────────────────────────────────────────────

def montar_texto_candidato(cand_id: int, dfs: dict[str, pd.DataFrame]) -> str | None:
    """
    dfs esperado (nomes exatos dos CSVs / tabelas do schema oficial):
      candidatos, candidato_habilidades_competencias, candidato_outras_habilidades,
      candidato_experiencias, candidato_formacoes, niveis_escolaridade,
      candidato_deficiencias, deficiencias, candidato_outras_deficiencias,
      candidato_adaptacoes, candidato_idiomas, idiomas
    """
    cand_df = dfs.get("candidatos", pd.DataFrame())
    cand = cand_df.loc[cand_df["cand_id_candidato"] == cand_id]
    if cand.empty:
        return None
    cand = cand.iloc[0]
    partes = []

    # 1. Habilidades estruturadas
    hab_df = dfs.get("candidato_habilidades_competencias", pd.DataFrame())
    hab = hab_df[hab_df["chco_id_candidato"] == cand_id]
    if not hab.empty:
        partes.append("Habilidades: " + "; ".join(
            hab["chco_descricao"].dropna().astype(str).unique()
        ))

    # 2. Habilidades em texto livre (campo único por candidato)
    oha_df = dfs.get("candidato_outras_habilidades", pd.DataFrame())
    oha = oha_df[oha_df["coha_id_candidato"] == cand_id]
    if not oha.empty:
        partes.append("Habilidades adicionais: " + oha.iloc[0]["coha_descricao"])

    # 3. Experiências — cargo + atividades
    exp_df = dfs.get("candidato_experiencias", pd.DataFrame())
    exp = exp_df[exp_df["cexp_id_candidato"] == cand_id]
    if not exp.empty:
        cargos = "; ".join(exp["cexp_cargo"].dropna().astype(str).tolist())
        partes.append(f"Experiência em: {cargos}")
        # Atividades trazem mais contexto semântico do que só o cargo
        atividades = " ".join(
            exp["cexp_atividades"].dropna().astype(str).tolist()
        )
        if atividades.strip():
            partes.append(f"Atividades: {atividades[:300]}")

    # 4. Formações — curso + área de formação + nível de escolaridade
    form_df = dfs.get("candidato_formacoes", pd.DataFrame())
    form = form_df[form_df["cfor_id_candidato"] == cand_id]
    if not form.empty:
        cursos = []
        nesc_df = dfs.get("niveis_escolaridade", pd.DataFrame())
        for _, row in form.iterrows():
            partes_curso = []
            if pd.notna(row.get("cfor_curso")):
                partes_curso.append(str(row["cfor_curso"]))
            if pd.notna(row.get("cfor_area_formacao")):
                partes_curso.append(str(row["cfor_area_formacao"]))
            if pd.notna(row.get("cfor_id_escolaridade")) and not nesc_df.empty:
                nivel = nesc_df[nesc_df["nesc_id_escolaridade"] == row["cfor_id_escolaridade"]]
                if not nivel.empty:
                    partes_curso.append(nivel.iloc[0]["nesc_descricao"])
            if partes_curso:
                cursos.append(" - ".join(partes_curso))
        if cursos:
            partes.append("Formação: " + "; ".join(cursos))

    # 5. Deficiências catalogadas + texto livre
    def_df  = dfs.get("candidato_deficiencias", pd.DataFrame())
    defi_df = dfs.get("deficiencias", pd.DataFrame())
    def_cand = def_df[def_df["cdef_id_candidato"] == cand_id]
    textos_def = []

    if not def_cand.empty and not defi_df.empty:
        merged = def_cand.merge(
            defi_df, left_on="cdef_id_deficiencia",
            right_on="defi_id_deficiencia", how="left"
        )
        textos_def += merged["defi_descricao"].dropna().astype(str).unique().tolist()

    ode_df = dfs.get("candidato_outras_deficiencias", pd.DataFrame())
    ode = ode_df[ode_df["code_id_candidato"] == cand_id]
    if not ode.empty:
        textos_def.append(ode.iloc[0]["code_descricao"])

    if textos_def:
        partes.append("PCD: " + "; ".join(textos_def))

    # 6. Adaptações necessárias (texto livre — relevante para vagas adaptadas)
    ada_df = dfs.get("candidato_adaptacoes", pd.DataFrame())
    ada = ada_df[ada_df["cada_id_candidato"] == cand_id]
    if not ada.empty:
        partes.append("Adaptações: " + ada.iloc[0]["cada_descricao"])

    # 7. Idiomas
    cidi_df  = dfs.get("candidato_idiomas", pd.DataFrame())
    idio_df  = dfs.get("idiomas", pd.DataFrame())
    cidi = cidi_df[cidi_df["cidi_id_candidato"] == cand_id]
    if not cidi.empty and not idio_df.empty:
        merged_i = cidi.merge(idio_df, left_on="cidi_id_idioma", right_on="idio_id_idioma", how="left")
        idiomas_txt = "; ".join(
            f"{r['idio_descricao']} ({r['cidi_proficiencia']})"
            for _, r in merged_i.iterrows()
            if pd.notna(r.get("idio_descricao"))
        )
        if idiomas_txt:
            partes.append(f"Idiomas: {idiomas_txt}")

    # 8. Local — menor peso semântico, fica no fim
    partes.append(f"Local: {cand['cand_cidade']} {cand['cand_estado']}")

    return " ".join(partes).strip()


def is_candidato_pcd(cand_id: int, dfs: dict[str, pd.DataFrame]) -> bool:
    """
    Candidato é PCD se:
      - tem flag cand_deficiencias=True, OU
      - tem registros em candidato_deficiencias, OU
      - tem texto em candidato_outras_deficiencias
    """
    cand_df = dfs.get("candidatos", pd.DataFrame())
    cand = cand_df[cand_df["cand_id_candidato"] == cand_id]
    if not cand.empty and cand.iloc[0].get("cand_deficiencias"):
        return True

    def_df = dfs.get("candidato_deficiencias", pd.DataFrame())
    if not def_df.empty and not def_df[def_df["cdef_id_candidato"] == cand_id].empty:
        return True

    ode_df = dfs.get("candidato_outras_deficiencias", pd.DataFrame())
    if not ode_df.empty and not ode_df[ode_df["code_id_candidato"] == cand_id].empty:
        return True

    return False


# ── Vaga ─────────────────────────────────────────────────────────────────────

def montar_texto_vaga(
    vaga_id: int,
    vaga_id_empresa: int,
    dfs: dict[str, pd.DataFrame],
) -> tuple[str, str | None]:
    """
    Retorna (texto_agregado, area_descricao).

    Assinatura mudou para receber (vaga_id, vaga_id_empresa) explicitamente
    por causa da PK composta.

    dfs esperado: vagas, vaga_habilidades_competencias, vaga_conhecimentos_tecnicos,
                  vaga_diferenciais, vaga_certificacoes, vaga_beneficios, vaga_funcoes,
                  areas_atuacao, vaga_outra_area_atuacao, vaga_modalidades, vaga_regimes,
                  niveis_escolaridade
    """
    vagas_df = dfs.get("vagas", pd.DataFrame())
    row_df = vagas_df[
        (vagas_df["vaga_id_vaga"] == vaga_id) &
        (vagas_df["vaga_id_empresa"] == vaga_id_empresa)
    ]
    if row_df.empty:
        return "", None
    row = row_df.iloc[0]

    partes    = []
    area_desc = None

    # 1. Título
    partes.append(f"Vaga: {row['vaga_titulo']}")

    # 2. Habilidades
    vhab_df = dfs.get("vaga_habilidades_competencias", pd.DataFrame())
    vhab = vhab_df[
        (vhab_df["vhab_id_vaga"] == vaga_id) &
        (vhab_df["vhab_id_empresa"] == vaga_id_empresa)
    ]
    if not vhab.empty:
        partes.append("Requisitos: " + "; ".join(
            vhab["vhab_descricao"].dropna().astype(str).unique()
        ))

    # 3. Conhecimentos técnicos
    vcon_df = dfs.get("vaga_conhecimentos_tecnicos", pd.DataFrame())
    vcon = vcon_df[
        (vcon_df["vcte_id_vaga"] == vaga_id) &
        (vcon_df["vcte_id_empresa"] == vaga_id_empresa)
    ]
    if not vcon.empty:
        partes.append("Conhecimentos: " + vcon.iloc[0]["vcte_descricao"])

    # 4. Diferenciais
    vdif_df = dfs.get("vaga_diferenciais", pd.DataFrame())
    vdif = vdif_df[
        (vdif_df["vdif_id_vaga"] == vaga_id) &
        (vdif_df["vdif_id_empresa"] == vaga_id_empresa)
    ]
    if not vdif.empty:
        partes.append("Diferenciais: " + vdif.iloc[0]["vdif_descricao"])

    # 5. Certificações
    vcer_df = dfs.get("vaga_certificacoes", pd.DataFrame())
    vcer = vcer_df[
        (vcer_df["vcer_id_vaga"] == vaga_id) &
        (vcer_df["vcer_id_empresa"] == vaga_id_empresa)
    ]
    if not vcer.empty:
        partes.append("Certificações: " + vcer.iloc[0]["vcer_descricao"])

    # 5b. Benefícios — texto adicional útil quando a vaga é muito curta
    vben_df = dfs.get("vaga_beneficios", pd.DataFrame())
    vben = vben_df[
        (vben_df.get("vben_id_vaga", pd.Series(dtype=int)) == vaga_id) &
        (vben_df.get("vben_id_empresa", pd.Series(dtype=int)) == vaga_id_empresa)
    ]
    if not vben.empty:
        partes.append("Benefícios: " + " ".join(vben["vben_descricao"].dropna().astype(str).unique()))

    # 6. Função + Área de atuação
    vfun_df = dfs.get("vaga_funcoes", pd.DataFrame())
    if pd.notna(row.get("vaga_funcao_id")):
        vfun = vfun_df[vfun_df["vfun_id_funcao"] == row["vaga_funcao_id"]]
        if not vfun.empty:
            partes.append(f"Função: {vfun.iloc[0]['vfun_descricao']}")
            area_df = dfs.get("areas_atuacao", pd.DataFrame())
            area = area_df[area_df["aatu_id_categoria"] == vfun.iloc[0]["vfun_id_area_atuacao"]]
            if not area.empty:
                area_desc = area.iloc[0]["aatu_descricao"]

    # 6b. Área de atuação em texto livre (quando não está no cadastro)
    voaa_df = dfs.get("vaga_outra_area_atuacao", pd.DataFrame())
    voaa = voaa_df[
        (voaa_df["voaa_id_vaga"] == vaga_id) &
        (voaa_df["voaa_id_empresa"] == vaga_id_empresa)
    ]
    if not voaa.empty:
        area_desc = area_desc or voaa.iloc[0]["voaa_descricao"]
        partes.append(f"Área: {voaa.iloc[0]['voaa_descricao']}")

    # 7. Escolaridade desejada
    nesc_df = dfs.get("niveis_escolaridade", pd.DataFrame())
    if pd.notna(row.get("vaga_escolaridade_desejada")) and not nesc_df.empty:
        nivel = nesc_df[nesc_df["nesc_id_escolaridade"] == row["vaga_escolaridade_desejada"]]
        if not nivel.empty:
            partes.append(f"Escolaridade desejada: {nivel.iloc[0]['nesc_descricao']}")

    # 8. Modalidade de trabalho (presencial / remoto / híbrido)
    vmod_df = dfs.get("vaga_modalidades", pd.DataFrame())
    if pd.notna(row.get("vaga_modalidade")) and not vmod_df.empty:
        mod = vmod_df[vmod_df["vmod_id"] == row["vaga_modalidade"]]
        if not mod.empty:
            partes.append(f"Modalidade: {mod.iloc[0]['vmod_descricao']}")

    # 8b. Regime de trabalho (quando a vaga for cadastrada com a tabela relacional)
    vreg_df = dfs.get("vaga_regimes", pd.DataFrame())
    if pd.notna(row.get("vaga_regime")) and not vreg_df.empty:
        reg = vreg_df[vreg_df["vreg_id_regime"] == row["vaga_regime"]]
        if not reg.empty:
            partes.append(f"Regime: {reg.iloc[0]['vreg_descricao']}")

    # 9. Descrição com expansão controlada para vagas curtas
    if pd.notna(row.get("vaga_descricao")):
        habilidades_txt = vhab["vhab_descricao"].dropna().astype(str).tolist() if not vhab.empty else []
        conhecimentos_txt = [str(vcon.iloc[0]["vcte_descricao"]) ] if not vcon.empty else []
        diferenciais_txt = [str(vdif.iloc[0]["vdif_descricao"]) ] if not vdif.empty else []
        certificacoes_txt = [str(vcer.iloc[0]["vcer_descricao"]) ] if not vcer.empty else []
        detalhes = _expandir_descricao_vaga_curta(
            titulo=str(row["vaga_titulo"]),
            funcao=str(vfun.iloc[0]["vfun_descricao"]) if pd.notna(row.get("vaga_funcao_id")) and not vfun.empty else "",
            area=area_desc,
            modalidade=str(mod.iloc[0]["vmod_descricao"]) if pd.notna(row.get("vaga_modalidade")) and not vmod_df.empty and not mod.empty else "",
            escolaridade_desejada=str(nivel.iloc[0]["nesc_descricao"]) if pd.notna(row.get("vaga_escolaridade_desejada")) and not nesc_df.empty and 'nivel' in locals() and not nivel.empty else "",
            regime=str(reg.iloc[0]["vreg_descricao"]) if pd.notna(row.get("vaga_regime")) and not vreg_df.empty and 'reg' in locals() and not reg.empty else "",
            habilidades=habilidades_txt,
            conhecimentos=conhecimentos_txt,
            diferenciais=diferenciais_txt,
            certificacoes=certificacoes_txt,
            descricao=str(row["vaga_descricao"]),
        )
        partes.append(f"Detalhes: {detalhes}")

    # 10. Local
    partes.append(f"Local: {row['vaga_cidade']} {row['vaga_estado']}")

    return " ".join(partes).strip(), area_desc


def is_vaga_pcd(titulo: str) -> bool:
    """Vaga é exclusiva PCD se o título contém 'pcd' (insensível a maiúsculas)."""
    return "pcd" in str(titulo).lower()
