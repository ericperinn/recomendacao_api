import csv, os

base = r'c:\Users\ericd\Documents\2026\tcc\recomendacao_api\dados'

def read_csv(name):
    p = os.path.join(base, name + '.csv')
    if not os.path.exists(p):
        return []
    with open(p, encoding='utf-8') as f:
        return list(csv.DictReader(f))

# Empresas
empr = read_csv('empresas')
empr_ids = set(int(r['empr_id_empresa']) for r in empr)
print(f'Empresas: {len(empr_ids)} registros, range: {min(empr_ids)}-{max(empr_ids)}')

# Vagas
vagas = read_csv('vagas')
vaga_empresa_ids = set(int(r['vaga_id_empresa']) for r in vagas)
print(f'Vagas: {len(vagas)} registros, empr_ids usados: range {min(vaga_empresa_ids)}-{max(vaga_empresa_ids)}')
missing_empr = vaga_empresa_ids - empr_ids
print(f'Empresa IDs em vagas mas NAO em empresas: {len(missing_empr)} -- {sorted(list(missing_empr))[:10]}')

# Candidatos
cands = read_csv('candidatos')
cand_ids = set(int(r['cand_id_candidato']) for r in cands)
print(f'\nCandidatos: {len(cand_ids)} registros')

# candidato_experiencias FK
exps = read_csv('candidato_experiencias')
exp_cand_ids = set(int(r['cexp_id_candidato']) for r in exps)
missing_cand_exp = exp_cand_ids - cand_ids
print(f'cand_ids em experiencias mas NAO em candidatos: {len(missing_cand_exp)}')

# candidato_deficiencias FK
cdefs = read_csv('candidato_deficiencias')
cdef_cand_ids = set(int(r['cdef_id_candidato']) for r in cdefs)
missing_cand_def = cdef_cand_ids - cand_ids
print(f'cand_ids em deficiencias mas NAO em candidatos: {len(missing_cand_def)}')

# deficiencias FK no candidato_deficiencias
defis = read_csv('deficiencias')
defi_ids = set(int(r['defi_id_deficiencia']) for r in defis)
cdef_defi_ids = set(int(r['cdef_id_deficiencia']) for r in cdefs)
missing_defi = cdef_defi_ids - defi_ids
print(f'defi_ids em candidato_deficiencias mas NAO em deficiencias: {len(missing_defi)} -- {sorted(list(missing_defi))[:10]}')

# vaga_funcoes FK to areas_atuacao
areas = read_csv('areas_atuacao')
area_ids = set(int(r['aatu_id_categoria']) for r in areas)
funcoes = read_csv('vaga_funcoes')
funcao_area_ids = set(int(r['vfun_id_area_atuacao']) for r in funcoes if r.get('vfun_id_area_atuacao'))
missing_area = funcao_area_ids - area_ids
print(f'\nvaga_funcoes area_ids NAO em areas_atuacao: {len(missing_area)} -- {sorted(list(missing_area))[:10]}')

# candidatos FK to generos
generos = read_csv('generos')
genero_ids = set(int(r['gene_id_genero']) for r in generos)
cand_genero_ids = set(int(r['cand_genero']) for r in cands if r.get('cand_genero'))
missing_genero = cand_genero_ids - genero_ids
print(f'candidatos genero_ids NAO em generos: {len(missing_genero)} -- {sorted(list(missing_genero))[:10]}')
print(f'Generos disponiveis: {[r["gene_id_genero"] for r in generos]}')

# vaga FK to funcoes
funcao_ids = set(int(r['vfun_id_funcao']) for r in funcoes)
vaga_funcao_ids = set(int(r['vaga_funcao_id']) for r in vagas if r.get('vaga_funcao_id'))
missing_funcao = vaga_funcao_ids - funcao_ids
print(f'\nvagas funcao_ids NAO em vaga_funcoes: {len(missing_funcao)} -- {sorted(list(missing_funcao))[:10]}')

# vaga FK to turnos
turnos = read_csv('turnos_trabalho')
turno_ids = set(int(r['tutr_id_turno']) for r in turnos)
vaga_turno_ids = set(int(r['vaga_turno']) for r in vagas if r.get('vaga_turno'))
missing_turno = vaga_turno_ids - turno_ids
print(f'vagas turno_ids NAO em turnos_trabalho: {len(missing_turno)} -- {sorted(list(missing_turno))[:10]}')

# vaga FK to modalidades
mods = read_csv('vaga_modalidades')
mod_ids = set(int(r['vmod_id']) for r in mods)
vaga_mod_ids = set(int(r['vaga_modalidade']) for r in vagas if r.get('vaga_modalidade'))
missing_mod = vaga_mod_ids - mod_ids
print(f'vagas modalidade_ids NAO em vaga_modalidades: {len(missing_mod)} -- {sorted(list(missing_mod))[:10]}')

# vaga FK to regimes
regimes = read_csv('vaga_regimes')
regime_ids = set(int(r['vreg_id_regime']) for r in regimes)
vaga_regime_ids = set(int(r['vaga_regime']) for r in vagas if r.get('vaga_regime'))
missing_regime = vaga_regime_ids - regime_ids
print(f'vagas regime_ids NAO em vaga_regimes: {len(missing_regime)} -- {sorted(list(missing_regime))[:10]}')

# candidato_idiomas FK to idiomas
idiomas = read_csv('idiomas')
idioma_ids = set(int(r['idio_id_idioma']) for r in idiomas)
cidis = read_csv('candidato_idiomas')
cidi_idioma_ids = set(int(r['cidi_id_idioma']) for r in cidis if r.get('cidi_id_idioma'))
missing_idioma = cidi_idioma_ids - idioma_ids
print(f'\ncandidato_idiomas idioma_ids NAO em idiomas: {len(missing_idioma)}')

# habilidades FK to candidatos
habs = read_csv('candidato_habilidades_competencias')
print(f'\nHabilidades candidatos: {len(habs)} registros (sample small file)')
print(f'Sample: {habs[:3] if habs else "empty"}')

# candidato_formacoes FK to escolaridade
escs = read_csv('niveis_escolaridade')
esc_ids = set(int(r['nesc_id_escolaridade']) for r in escs)
forms = read_csv('candidato_formacoes')
form_esc_ids = set(int(r['cfor_id_escolaridade']) for r in forms if r.get('cfor_id_escolaridade'))
missing_esc = form_esc_ids - esc_ids
print(f'\nformacoes escolaridade_ids NAO em niveis_escolaridade: {len(missing_esc)}')

print('\nDone.')
