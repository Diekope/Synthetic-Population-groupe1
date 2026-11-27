"""Référence exacte (plan C) : max nombre de personas distinctes, marges exactes.
Usage : python reference_cpsat.py contraintes.csv [temps_s] [population_initiale.csv|-] [sortie.csv]"""
import collections
import itertools
import sys

import pandas as pd
from ortools.sat.python import cp_model

d = pd.read_csv(sys.argv[1]); limit = float(sys.argv[2]) if len(sys.argv) > 2 else 120
tables = {}
for r in d.itertuples():
    kv = dict(p.split('=') for p in r.cell.split(';'))
    tables.setdefault(r.table_id, {})[tuple(sorted(kv.items()))] = r.target
attrs = sorted({a for t in tables.values() for c in t for a, _ in c}, key=lambda s: int(s[1:]))
vals = {a: sorted({v for t in tables.values() for c in t for b, v in c if b == a}) for a in attrs}
N = sum(next(iter(tables.values())).values())
scopes = {tid: [a for a, _ in next(iter(t))] for tid, t in tables.items()}
active = []
for x in itertools.product(*[vals[a] for a in attrs]):
    xd = dict(zip(attrs, x)); u = N
    for tid, t in tables.items():
        u = min(u, t.get(tuple((a, xd[a]) for a in scopes[tid]), 0))
        if u == 0: break
    if u > 0: active.append((x, u))
print("personas actives :", len(active))
m = cp_model.CpModel(); n = []; y = []
cells = collections.defaultdict(list)
for k, (x, u) in enumerate(active):
    n.append(m.NewIntVar(0, u, f"n{k}")); y.append(m.NewBoolVar(f"y{k}"))
    m.Add(n[k] >= y[k])
    xd = dict(zip(attrs, x))
    for tid in tables: cells[(tid, tuple((a, xd[a]) for a in scopes[tid]))].append(k)
for tid, t in tables.items():
    for c, tg in t.items():
        m.Add(sum(n[k] for k in cells.get((tid, c), [])) == tg)
if len(sys.argv) > 3 and sys.argv[3] != "-":   # solution initiale (hint) issue du greedy Java
    pop = pd.read_csv(sys.argv[3]); cnt = collections.Counter(tuple(r[a] for a in attrs) for _, r in pop.iterrows())
    for k, (x, u) in enumerate(active):
        m.AddHint(n[k], cnt.get(x, 0)); m.AddHint(y[k], 1 if cnt.get(x, 0) > 0 else 0)
    print("hint : distinctes =", len(cnt))
m.Maximize(sum(y))
s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = limit; s.parameters.num_workers = 8
st = s.Solve(m)
print(s.StatusName(st), "distinctes =", int(s.ObjectiveValue()), "borne sup =", int(s.BestObjectiveBound()))
if len(sys.argv) > 4 and st in (cp_model.OPTIMAL, cp_model.FEASIBLE):   # population trouvée, même format que GreedyMain
    rows = [x for k, (x, u) in enumerate(active) for _ in range(s.Value(n[k]))]
    pd.DataFrame(rows, columns=attrs).rename_axis("id").to_csv(sys.argv[4])
    print("écrit :", sys.argv[4], len(rows), "individus")
