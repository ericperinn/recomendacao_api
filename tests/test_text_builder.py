"""
Testes unitários para text_builder — garante que a lógica de montagem de texto
(transcrição fiel do notebook) produz os resultados esperados.
"""
import pandas as pd
import pytest

from app.services.text_builder import (
    limpar_texto,
    montar_texto_candidato,
    montar_texto_vaga,
    is_candidato_pcd,
    is_vaga_pcd,
)


# ── Fixtures de DataFrames mínimos ────────────────────────────────────────────

@pytest.fixture
def dfs_candidato():
    return {
        "candidatos": pd.DataFrame([{
            "cand_id_candidato": 1,
            "cand_nome":   "Eric Perin",
            "cand_cidade": "Manaus",
            "cand_estado": "AM",
            "cand_deficiencias": False,
        }]),
        "candidato_habilidades_competencias": pd.DataFrame([
            {"chco_id_candidato": 1, "chco_descricao": "Python"},
            {"chco_id_candidato": 1, "chco_descricao": "FastAPI"},
        ]),
        "candidato_outras_habilidades": pd.DataFrame(
            columns=["coha_id_candidato", "coha_descricao"]
        ),
        "candidato_experiencias": pd.DataFrame([
            {
                "cexp_id_candidato": 1,
                "cexp_cargo": "Desenvolvedor Full Stack",
                "cexp_atividades": "Desenvolvimento de APIs REST.",
                "cexp_emprego_atual": False,
            },
        ]),
        "candidato_formacoes": pd.DataFrame([
            {
                "cfor_id_candidato":    1,
                "cfor_curso":           "Engenharia de Computação",
                "cfor_area_formacao":   "Tecnologia",
                "cfor_id_escolaridade": 1,
                "cfor_situacao":        "Concluído",
                "cfor_instituicao":     "UEA",
                "cfor_ano_inicio":      "2018",
            },
        ]),
        "niveis_escolaridade": pd.DataFrame([
            {"nesc_id_escolaridade": 1, "nesc_descricao": "Superior Completo"},
        ]),
        "candidato_deficiencias": pd.DataFrame(
            columns=["cdef_id_candidato", "cdef_id_deficiencia"]
        ),
        "deficiencias": pd.DataFrame(
            columns=["defi_id_deficiencia", "defi_descricao"]
        ),
        "candidato_outras_deficiencias": pd.DataFrame(
            columns=["code_id_candidato", "code_descricao"]
        ),
        "candidato_adaptacoes": pd.DataFrame(
            columns=["cada_id_candidato", "cada_descricao"]
        ),
        "candidato_idiomas": pd.DataFrame(
            columns=["cidi_id_candidato", "cidi_id_idioma", "cidi_proficiencia"]
        ),
        "idiomas": pd.DataFrame(
            columns=["idio_id_idioma", "idio_descricao"]
        ),
    }


@pytest.fixture
def dfs_candidato_pcd():
    return {
        "candidatos": pd.DataFrame([{
            "cand_id_candidato": 2,
            "cand_nome":   "Igor Ramos",
            "cand_cidade": "Manaus",
            "cand_estado": "AM",
            "cand_deficiencias": True,
        }]),
        "candidato_habilidades_competencias": pd.DataFrame(
            columns=["chco_id_candidato", "chco_descricao"]
        ),
        "candidato_outras_habilidades": pd.DataFrame(
            columns=["coha_id_candidato", "coha_descricao"]
        ),
        "candidato_experiencias": pd.DataFrame([
            {
                "cexp_id_candidato": 2,
                "cexp_cargo": "Motorista",
                "cexp_atividades": "Transporte de passageiros.",
                "cexp_emprego_atual": True,
            },
        ]),
        "candidato_formacoes": pd.DataFrame(
            columns=["cfor_id_candidato", "cfor_curso", "cfor_area_formacao",
                     "cfor_id_escolaridade", "cfor_situacao", "cfor_instituicao", "cfor_ano_inicio"]
        ),
        "niveis_escolaridade": pd.DataFrame(
            columns=["nesc_id_escolaridade", "nesc_descricao"]
        ),
        "candidato_deficiencias": pd.DataFrame([
            {"cdef_id_candidato": 2, "cdef_id_deficiencia": 1},
        ]),
        "deficiencias": pd.DataFrame([
            {"defi_id_deficiencia": 1, "defi_descricao": "Mobilidade reduzida"},
        ]),
        "candidato_outras_deficiencias": pd.DataFrame(
            columns=["code_id_candidato", "code_descricao"]
        ),
        "candidato_adaptacoes": pd.DataFrame(
            columns=["cada_id_candidato", "cada_descricao"]
        ),
        "candidato_idiomas": pd.DataFrame(
            columns=["cidi_id_candidato", "cidi_id_idioma", "cidi_proficiencia"]
        ),
        "idiomas": pd.DataFrame(
            columns=["idio_id_idioma", "idio_descricao"]
        ),
    }


@pytest.fixture
def dfs_vaga():
    return {
        "vagas": pd.DataFrame([{
            "vaga_id_vaga":    100,
            "vaga_id_empresa": 1,
            "vaga_titulo":     "Dev Full Stack Python",
            "vaga_descricao":  "Vaga para desenvolvedor Python.",
            "vaga_cidade":     "Manaus",
            "vaga_estado":     "AM",
            "vaga_funcao_id":  1,
            "vaga_modalidade": 1,
            "vaga_escolaridade_desejada": 1,
        }]),
        "vaga_habilidades_competencias": pd.DataFrame([
            {"vhab_id_vaga": 100, "vhab_id_empresa": 1, "vhab_descricao": "Python"},
            {"vhab_id_vaga": 100, "vhab_id_empresa": 1, "vhab_descricao": "SQL"},
        ]),
        "vaga_conhecimentos_tecnicos": pd.DataFrame(
            columns=["vcte_id_vaga", "vcte_id_empresa", "vcte_descricao"]
        ),
        "vaga_diferenciais": pd.DataFrame(
            columns=["vdif_id_vaga", "vdif_id_empresa", "vdif_descricao"]
        ),
        "vaga_certificacoes": pd.DataFrame(
            columns=["vcer_id_vaga", "vcer_id_empresa", "vcer_descricao"]
        ),
        "vaga_outra_area_atuacao": pd.DataFrame(
            columns=["voaa_id_vaga", "voaa_id_empresa", "voaa_descricao"]
        ),
        "vaga_funcoes": pd.DataFrame([
            {"vfun_id_funcao": 1, "vfun_descricao": "Desenvolvedor", "vfun_id_area_atuacao": 10},
        ]),
        "areas_atuacao": pd.DataFrame([
            {"aatu_id_categoria": 10, "aatu_descricao": "Tecnologia da Informação"},
        ]),
        "vaga_modalidades": pd.DataFrame([
            {"vmod_id": 1, "vmod_descricao": "Presencial"},
        ]),
        "niveis_escolaridade": pd.DataFrame([
            {"nesc_id_escolaridade": 1, "nesc_descricao": "Superior Completo"},
        ]),
    }


# ── Testes de limpeza de texto ────────────────────────────────────────────────

def test_limpar_texto_remove_stopwords():
    texto = limpar_texto("experiência em comunicação e organização")
    # "experiência", "comunicação" e "organização" estão nas stopwords customizadas
    assert "experiência" not in texto
    assert "comunicação" not in texto

def test_limpar_texto_preserva_termos_tecnicos():
    texto = limpar_texto("python fastapi nodejs typescript")
    assert "python" in texto
    assert "fastapi" in texto

def test_limpar_texto_preserva_nivel():
    # júnior, pleno, sênior NÃO estão nas stopwords — devem ser preservados
    texto = limpar_texto("desenvolvedor júnior pleno sênior")
    assert "júnior" in texto
    assert "pleno" in texto

def test_limpar_texto_remove_pontuacao():
    texto = limpar_texto("Python! FastAPI, SQL.")
    assert "!" not in texto
    assert "," not in texto


# ── Testes de montagem de texto — candidato ───────────────────────────────────

def test_montar_texto_candidato_contem_habilidades(dfs_candidato):
    texto = montar_texto_candidato(1, dfs_candidato)
    assert texto is not None
    assert "Python" in texto
    assert "FastAPI" in texto

def test_montar_texto_candidato_contem_experiencia(dfs_candidato):
    texto = montar_texto_candidato(1, dfs_candidato)
    assert "Desenvolvedor Full Stack" in texto

def test_montar_texto_candidato_contem_formacao(dfs_candidato):
    texto = montar_texto_candidato(1, dfs_candidato)
    assert "Engenharia de Computação" in texto

def test_montar_texto_candidato_pcd_contem_pcd(dfs_candidato_pcd):
    texto = montar_texto_candidato(2, dfs_candidato_pcd)
    assert "PCD" in texto
    assert "Mobilidade reduzida" in texto

def test_montar_texto_candidato_inexistente(dfs_candidato):
    resultado = montar_texto_candidato(9999, dfs_candidato)
    assert resultado is None

def test_is_candidato_pcd_falso(dfs_candidato):
    assert is_candidato_pcd(1, dfs_candidato) is False

def test_is_candidato_pcd_verdadeiro(dfs_candidato_pcd):
    assert is_candidato_pcd(2, dfs_candidato_pcd) is True


# ── Testes de montagem de texto — vaga ────────────────────────────────────────

def test_montar_texto_vaga_contem_titulo(dfs_vaga):
    texto, area = montar_texto_vaga(100, 1, dfs_vaga)
    assert "Dev Full Stack Python" in texto

def test_montar_texto_vaga_contem_habilidades(dfs_vaga):
    texto, _ = montar_texto_vaga(100, 1, dfs_vaga)
    assert "Python" in texto
    assert "SQL" in texto

def test_montar_texto_vaga_retorna_area(dfs_vaga):
    _, area = montar_texto_vaga(100, 1, dfs_vaga)
    assert area == "Tecnologia da Informação"

def test_is_vaga_pcd_false(dfs_vaga):
    assert is_vaga_pcd("Dev Full Stack Python") is False

def test_is_vaga_pcd_true():
    assert is_vaga_pcd("Vaga PCD - Analista Júnior") is True
