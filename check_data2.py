import csv, os

base = r"c:\Users\ericd\Documents\2026\tcc\recomendacao_api\dados"

with open(os.path.join(base, "candidatos.csv"), encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print("=== Sample candidatos rows ===")
for i, r in enumerate(rows[:5]):
    print(
        f"data_cad={repr(r['cand_data_cad'])}, defic={repr(r['cand_deficiencias'])}, genero={repr(r['cand_genero'])}"
    )

# Find critical empties
for field in [
    "cand_data_cad",
    "cand_deficiencias",
    "cand_adaptacoes",
    "cand_laudo_carteira",
    "cand_ativo",
    "cand_hash_senha",
]:
    empties = [r for r in rows if r.get(field, "") == ""]
    print(f"Empty {field}: {len(empties)}")

# Check if date values look right
from datetime import datetime

bad_dates = []
for r in rows:
    d = r.get("cand_data_cad", "")
    if d:
        try:
            datetime.strptime(d, "%Y-%m-%d")
        except:
            bad_dates.append((r.get("cand_id_candidato"), d))
print(f"Bad cand_data_cad values: {len(bad_dates)} -> {bad_dates[:5]}")

# Check cand_data_nascto
bad_dates2 = []
for r in rows:
    d = r.get("cand_data_nascto", "")
    if d:
        try:
            datetime.strptime(d, "%Y-%m-%d")
        except:
            bad_dates2.append((r.get("cand_id_candidato"), repr(d)))
print(f"Bad cand_data_nascto: {len(bad_dates2)} -> {bad_dates2[:5]}")

# Check the actual error - asyncpg sends add_all at commit
# The error shows dates like '2025-10-06' which look fine
# Maybe the issue is with boolean False being sent as Python bool for a String column?
# Check cand_genero values
genero_vals = set(r.get("cand_genero", "") for r in rows)
print(f"cand_genero unique values: {sorted(genero_vals)[:20]}")

# Check candidato_experiencias for date issues
with open(os.path.join(base, "candidato_experiencias.csv"), encoding="utf-8") as f:
    exps = list(csv.DictReader(f))

bad_exp_dates = []
for r in exps:
    for field in ["cexp_data_inicio", "cexp_data_fim"]:
        d = r.get(field, "")
        if d:
            try:
                datetime.strptime(d, "%Y-%m-%d")
            except:
                bad_exp_dates.append((field, repr(d)))
print(f"Bad experience dates: {len(bad_exp_dates)} -> {bad_exp_dates[:10]}")

# Check cexp_emprego_atual
emprego_vals = set(r.get("cexp_emprego_atual", "") for r in exps)
print(f"cexp_emprego_atual values: {sorted(emprego_vals)}")

print("Done.")
