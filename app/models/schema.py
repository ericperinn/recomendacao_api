"""
ORM espelhando o schema Prisma do sistema original (PostgreSQL).
Apenas as tabelas necessárias para montar o texto semântico são mapeadas aqui.
As tabelas de embedding (candidato_embeddings, vaga_embeddings) são próprias
desta API e não existem no sistema original.

ATENÇÃO: vagas tem chave composta (vaga_id_vaga, vaga_id_empresa).
Todas as tabelas filhas de vagas também carregam vaga_id_empresa.
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    Text,
    ForeignKey,
    LargeBinary,
    DateTime,
    Date,
    PrimaryKeyConstraint,
    ForeignKeyConstraint,
    func,
)
from app.core.database import BaseDados, BaseEmbeddings


# ── Tabelas de lookup ─────────────────────────────────────────────────────────


class Idioma(BaseDados):
    __tablename__ = "idiomas"
    idio_id_idioma = Column(Integer, primary_key=True, autoincrement=True)
    idio_descricao = Column(String(30), unique=True)


class Deficiencia(BaseDados):
    __tablename__ = "deficiencias"
    defi_id_deficiencia = Column(Integer, primary_key=True, autoincrement=True)
    defi_descricao = Column(String(150), unique=True, nullable=False)


class NivelEscolaridade(BaseDados):
    __tablename__ = "niveis_escolaridade"
    nesc_id_escolaridade = Column(Integer, primary_key=True, autoincrement=True)
    nesc_descricao = Column(String(30), unique=True, nullable=False)


class AreaAtuacao(BaseDados):
    __tablename__ = "areas_atuacao"
    aatu_id_categoria = Column(Integer, primary_key=True, autoincrement=True)
    aatu_descricao = Column(String, nullable=False)


class VagaFuncao(BaseDados):
    __tablename__ = "vaga_funcoes"
    vfun_id_funcao = Column(Integer, primary_key=True, autoincrement=True)
    vfun_descricao = Column(String, nullable=False)
    vfun_id_area_atuacao = Column(
        Integer, ForeignKey("areas_atuacao.aatu_id_categoria")
    )


class VagaModalidade(BaseDados):
    __tablename__ = "vaga_modalidades"
    vmod_id = Column(Integer, primary_key=True, autoincrement=True)
    vmod_descricao = Column(String(10), unique=True, nullable=False)


class VagaRegime(BaseDados):
    __tablename__ = "vaga_regimes"
    vreg_id_regime = Column(Integer, primary_key=True, autoincrement=True)
    vreg_descricao = Column(String(20), unique=True, nullable=False)


class TurnoTrabalho(BaseDados):
    __tablename__ = "turnos_trabalho"
    tutr_id_turno = Column(Integer, primary_key=True, autoincrement=True)
    tutr_descricao = Column(String(25), unique=True, nullable=False)


# ── Candidatos ───────────────────────────────────────────────────────────────


class Candidato(BaseDados):
    __tablename__ = "candidatos"

    cand_id_candidato = Column(Integer, primary_key=True, autoincrement=True)
    cand_nome = Column(String(150), nullable=False)
    cand_cpf = Column(String(11), unique=True, nullable=False)
    cand_email = Column(String(150), unique=True, nullable=False)
    cand_cidade = Column(String(50), nullable=False)
    cand_estado = Column(String(20), nullable=False)
    cand_cep = Column(String(8))
    cand_rua = Column(String(150))
    cand_bairro = Column(String(60))
    cand_numero = Column(String(20))
    cand_complemento = Column(String(150))
    cand_deficiencias = Column(Boolean, nullable=False)
    cand_adaptacoes = Column(Boolean, nullable=False)
    cand_laudo_carteira = Column(Boolean, nullable=False)
    cand_ativo = Column(Boolean, nullable=False)
    cand_data_cad = Column(Date, nullable=False)
    cand_hash_senha = Column(String(128), nullable=False)
    cand_data_nascto = Column(Date)
    cand_disp_mudanca = Column(Boolean)
    cand_disp_trab_remoto = Column(Boolean)
    cand_disp_viagens = Column(Boolean)
    cand_genero = Column(Integer, ForeignKey("generos.gene_id_genero"))


class Genero(BaseDados):
    __tablename__ = "generos"
    gene_id_genero = Column(Integer, primary_key=True, autoincrement=True)
    gene_descricao = Column(String(60), unique=True, nullable=False)


class CandidatoHabilidade(BaseDados):
    __tablename__ = "candidato_habilidades_competencias"
    __table_args__ = (
        PrimaryKeyConstraint("chco_id_habilidade_competencia", "chco_id_candidato"),
    )
    chco_id_habilidade_competencia = Column(Integer, autoincrement=True)
    chco_id_candidato = Column(
        Integer, ForeignKey("candidatos.cand_id_candidato", ondelete="CASCADE")
    )
    chco_descricao = Column(String, nullable=False)


class CandidatoOutraHabilidade(BaseDados):
    """Habilidades em texto livre (campo único por candidato)."""

    __tablename__ = "candidato_outras_habilidades"
    coha_id_candidato = Column(
        Integer,
        ForeignKey("candidatos.cand_id_candidato", ondelete="CASCADE"),
        primary_key=True,
    )
    coha_descricao = Column(String, nullable=False)


class CandidatoExperiencia(BaseDados):
    __tablename__ = "candidato_experiencias"
    cexp_id_experiencia = Column(Integer, primary_key=True, autoincrement=True)
    cexp_id_candidato = Column(
        Integer,
        ForeignKey("candidatos.cand_id_candidato", ondelete="CASCADE"),
        nullable=False,
    )
    cexp_cargo = Column(String(60), nullable=False)
    cexp_nome_empresa = Column(String(255), nullable=False)
    cexp_atividades = Column(String, nullable=False)
    cexp_data_inicio = Column(Date, nullable=False)
    cexp_data_fim = Column(Date)
    cexp_emprego_atual = Column(Boolean, nullable=False)


class CandidatoFormacao(BaseDados):
    __tablename__ = "candidato_formacoes"
    cfor_id_formacao = Column(Integer, primary_key=True, autoincrement=True)
    cfor_id_candidato = Column(
        Integer, ForeignKey("candidatos.cand_id_candidato", ondelete="CASCADE")
    )
    cfor_id_escolaridade = Column(
        Integer, ForeignKey("niveis_escolaridade.nesc_id_escolaridade")
    )
    cfor_curso = Column(String(150))
    cfor_instituicao = Column(String(150), nullable=False)
    cfor_situacao = Column(String(15), nullable=False)
    cfor_area_formacao = Column(String(150))
    cfor_ano_inicio = Column(String(4), nullable=False)
    cfor_ano_conclusao = Column(String(4))


class CandidatoDeficiencia(BaseDados):
    __tablename__ = "candidato_deficiencias"
    __table_args__ = (PrimaryKeyConstraint("cdef_id_candidato", "cdef_id_deficiencia"),)
    cdef_id_candidato = Column(
        Integer, ForeignKey("candidatos.cand_id_candidato", ondelete="CASCADE")
    )
    cdef_id_deficiencia = Column(
        Integer, ForeignKey("deficiencias.defi_id_deficiencia")
    )


class CandidatoOutraDeficiencia(BaseDados):
    """Deficiência descrita em texto livre (não está na tabela de lookup)."""

    __tablename__ = "candidato_outras_deficiencias"
    code_id_candidato = Column(
        Integer,
        ForeignKey("candidatos.cand_id_candidato", ondelete="CASCADE"),
        primary_key=True,
    )
    code_descricao = Column(String, nullable=False)


class CandidatoAdaptacao(BaseDados):
    """Adaptações necessárias — texto livre, campo único por candidato."""

    __tablename__ = "candidato_adaptacoes"
    cada_id_candidato = Column(
        Integer,
        ForeignKey("candidatos.cand_id_candidato", ondelete="CASCADE"),
        primary_key=True,
    )
    cada_descricao = Column(String, nullable=False)


class CandidatoIdioma(BaseDados):
    __tablename__ = "candidato_idiomas"
    __table_args__ = (PrimaryKeyConstraint("cidi_id_idioma", "cidi_id_candidato"),)
    cidi_id_idioma = Column(Integer, ForeignKey("idiomas.idio_id_idioma"))
    cidi_id_candidato = Column(
        Integer, ForeignKey("candidatos.cand_id_candidato", ondelete="CASCADE")
    )
    cidi_proficiencia = Column(String(15), nullable=False)


# ── Vagas ─────────────────────────────────────────────────────────────────────
# ATENÇÃO: PK composta (vaga_id_vaga, vaga_id_empresa)


class Vaga(BaseDados):
    __tablename__ = "vagas"
    __table_args__ = (PrimaryKeyConstraint("vaga_id_vaga", "vaga_id_empresa"),)
    vaga_id_vaga = Column(Integer, autoincrement=True)
    vaga_id_empresa = Column(Integer, ForeignKey("empresas.empr_id_empresa"))
    vaga_titulo = Column(String(100), nullable=False)
    vaga_descricao = Column(String, nullable=False)
    vaga_cidade = Column(String(50), nullable=False)
    vaga_estado = Column(String(20), nullable=False)
    vaga_status = Column(String(15), nullable=False)
    vaga_data_cad = Column(Date, nullable=False)
    vaga_tempo_experiencia = Column(Integer, nullable=False)
    vaga_funcao_id = Column(Integer, ForeignKey("vaga_funcoes.vfun_id_funcao"))
    vaga_turno = Column(Integer, ForeignKey("turnos_trabalho.tutr_id_turno"))
    vaga_modalidade = Column(Integer, ForeignKey("vaga_modalidades.vmod_id"))
    vaga_regime = Column(Integer, ForeignKey("vaga_regimes.vreg_id_regime"))
    vaga_escolaridade_minima = Column(
        Integer, ForeignKey("niveis_escolaridade.nesc_id_escolaridade")
    )
    vaga_escolaridade_desejada = Column(
        Integer, ForeignKey("niveis_escolaridade.nesc_id_escolaridade")
    )


class Empresa(BaseDados):
    __tablename__ = "empresas"
    empr_id_empresa = Column(Integer, primary_key=True, autoincrement=True)
    empr_nome = Column(String(150), nullable=False)
    empr_cidade = Column(String(50), nullable=False)
    empr_estado = Column(String(20), nullable=False)
    empr_cnpj = Column(String(14), unique=True, nullable=False)
    empr_email = Column(String(100), unique=True, nullable=False)
    empr_hash_senha = Column(String(128), nullable=False)
    empr_rua = Column(String(150))
    empr_bairro = Column(String(60))
    empr_cep = Column(String(8))
    empr_numero = Column(String(20))
    empr_complemento = Column(String(100))
    empr_descricao = Column(String)
    empr_site = Column(String(255))
    empr_ativo = Column(Boolean, nullable=False)
    empr_logo = Column(String(255))


class VagaHabilidade(BaseDados):
    __tablename__ = "vaga_habilidades_competencias"
    __table_args__ = (
        PrimaryKeyConstraint(
            "vhab_id_habilidade_competencias", "vhab_id_vaga", "vhab_id_empresa"
        ),
        ForeignKeyConstraint(
            ["vhab_id_vaga", "vhab_id_empresa"],
            ["vagas.vaga_id_vaga", "vagas.vaga_id_empresa"],
            ondelete="CASCADE",
        ),
    )
    vhab_id_habilidade_competencias = Column(Integer, autoincrement=True)
    vhab_id_vaga = Column(Integer, nullable=False)
    vhab_id_empresa = Column(Integer, nullable=False)
    vhab_descricao = Column(String, nullable=False)


class VagaConhecimento(BaseDados):
    __tablename__ = "vaga_conhecimentos_tecnicos"
    __table_args__ = (
        PrimaryKeyConstraint("vcte_id_vaga", "vcte_id_empresa"),
        ForeignKeyConstraint(
            ["vcte_id_vaga", "vcte_id_empresa"],
            ["vagas.vaga_id_vaga", "vagas.vaga_id_empresa"],
            ondelete="CASCADE",
        ),
    )
    vcte_id_vaga = Column(Integer, nullable=False)
    vcte_id_empresa = Column(Integer, nullable=False)
    vcte_descricao = Column(String, nullable=False)


class VagaDiferencial(BaseDados):
    __tablename__ = "vaga_diferenciais"
    __table_args__ = (
        PrimaryKeyConstraint("vdif_id_vaga", "vdif_id_empresa"),
        ForeignKeyConstraint(
            ["vdif_id_vaga", "vdif_id_empresa"],
            ["vagas.vaga_id_vaga", "vagas.vaga_id_empresa"],
            ondelete="CASCADE",
        ),
    )
    vdif_id_vaga = Column(Integer, nullable=False)
    vdif_id_empresa = Column(Integer, nullable=False)
    vdif_descricao = Column(String, nullable=False)


class VagaCertificacao(BaseDados):
    __tablename__ = "vaga_certificacoes"
    __table_args__ = (
        PrimaryKeyConstraint("vcer_id_vaga", "vcer_id_empresa"),
        ForeignKeyConstraint(
            ["vcer_id_vaga", "vcer_id_empresa"],
            ["vagas.vaga_id_vaga", "vagas.vaga_id_empresa"],
            ondelete="CASCADE",
        ),
    )
    vcer_id_vaga = Column(Integer, nullable=False)
    vcer_id_empresa = Column(Integer, nullable=False)
    vcer_descricao = Column(String, nullable=False)


class VagaOutraAreaAtuacao(BaseDados):
    """Área de atuação em texto livre quando não existe no cadastro."""

    __tablename__ = "vaga_outra_area_atuacao"
    __table_args__ = (
        PrimaryKeyConstraint("voaa_id_vaga", "voaa_id_empresa"),
        ForeignKeyConstraint(
            ["voaa_id_vaga", "voaa_id_empresa"],
            ["vagas.vaga_id_vaga", "vagas.vaga_id_empresa"],
            ondelete="CASCADE",
        ),
    )
    voaa_id_vaga = Column(Integer, nullable=False)
    voaa_id_empresa = Column(Integer, nullable=False)
    voaa_descricao = Column(String, nullable=False)


class VagaOutroRegime(BaseDados):
    __tablename__ = "vaga_outro_regime"
    __table_args__ = (
        PrimaryKeyConstraint("vore_id_vaga", "vore_id_empresa"),
        ForeignKeyConstraint(
            ["vore_id_vaga", "vore_id_empresa"],
            ["vagas.vaga_id_vaga", "vagas.vaga_id_empresa"],
            ondelete="CASCADE",
        ),
    )
    vore_id_vaga = Column(Integer, nullable=False)
    vore_id_empresa = Column(Integer, nullable=False)
    vore_descricao = Column(String, nullable=False)


# ── Tabelas próprias da API (não existem no sistema original) ─────────────────


class CandidatoEmbedding(BaseEmbeddings):
    """
    Embedding pré-calculado por candidato.
    Regerado via POST /admin/recalcular-embeddings quando o modelo muda.
    """

    __tablename__ = "candidato_embeddings"

    cand_id = Column(Integer, primary_key=True)
    texto_limpo = Column(Text)
    is_pcd = Column(Boolean, default=False)
    embedding = Column(LargeBinary)  # np.ndarray float32 serializado (.tobytes())
    modelo_usado = Column(String(200))
    atualizado_em = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class VagaEmbedding(BaseEmbeddings):
    """
    Embedding pré-calculado por vaga.
    PK simples aqui (vaga_id apenas) — vaga_id_empresa fica como coluna auxiliar
    para rastreabilidade, sem ser FK composta (simplifica queries de recomendação).
    """

    __tablename__ = "vaga_embeddings"

    vaga_id = Column(Integer, primary_key=True)  # = vaga_id_vaga
    vaga_id_empresa = Column(Integer, nullable=False)  # rastreabilidade
    vaga_titulo = Column(String(100))
    vaga_area = Column(String(255))
    texto_limpo = Column(Text)
    is_pcd_exclusive = Column(Boolean, default=False)
    embedding = Column(LargeBinary)
    modelo_usado = Column(String(200))
    atualizado_em = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
