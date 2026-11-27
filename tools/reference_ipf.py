"""Référence de réalisme : loi d'entropie maximale P* sous les tables (IPF sur l'espace complet des personas),
et statistiques attendues d'un échantillon i.i.d. de N individus tiré selon P* (distinctes, paires à distance 0, 1, 2).

P* est la loi la moins structurée compatible avec les marges : un échantillon de P* est une population « typique »
quand on ne connaît que les tables. Ses ressemblances servent de niveau de référence pour les populations construites.
Un échantillon i.i.d. ne respecte les marges qu'en moyenne (pas exactement).

Usage : python tools/reference_ipf.py constraint_cells.csv     (numpy requis)
"""
import collections
import csv
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

rows = list(csv.DictReader(open(sys.argv[1], encoding='utf-8')))
attrs, vals = [], collections.defaultdict(list)
for r in rows:
    for kv in r['cell'].split(';'):
        a, v = kv.split('=')
        if a not in attrs: attrs.append(a)
        if v not in vals[a]: vals[a].append(v)
attrs.sort(key=lambda s: int(s[1:]))
for a in attrs: vals[a].sort(key=lambda s: int(s[1:]))
K = len(attrs); dom = [len(vals[a]) for a in attrs]
tables = collections.defaultdict(list)
for r in rows: tables[r['table_id']].append(r)
T = []
for tid, cells in tables.items():
    scope = [attrs.index(a) for a in cells[0]['scope'].split('|')]
    tgt = np.zeros([dom[b] for b in scope])
    for r in cells:
        idx = tuple(vals[attrs[b]].index(kv.split('=')[1]) for b, kv in zip(scope, r['cell'].split(';')))
        tgt[idx] = int(r['target'])
    T.append((scope, tgt))
N = int(T[0][1].sum())

p = np.ones(dom); p /= p.sum()
for it in range(500):
    err = 0
    for scope, tgt in T:
        other = tuple(b for b in range(K) if b not in scope)
        m = p.sum(axis=other) * N                       # axes restants dans l'ordre croissant
        order = sorted(scope)
        t = np.transpose(tgt, [scope.index(b) for b in order])
        err = max(err, np.abs(m - t).max())
        ratio = np.divide(t, m, out=np.zeros_like(t), where=m > 0)
        shape = [dom[b] if b in scope else 1 for b in range(K)]
        p *= ratio.reshape(shape)
    if err < 1e-9: break
print(f"IPF : {it + 1} itérations, écart max sur les marges {err:.2e}, personas de poids > 0 : {(p > 1e-15).sum()}")

def same(p, q): return float((p * q).sum())
P0 = same(p, p)
P1 = sum(same(p, np.roll(p, s, axis=a)) for a in range(K) for s in range(1, dom[a]))
P2 = 0.0
for a in range(K):
    for b in range(a + 1, K):
        for s in range(1, dom[a]):
            for u in range(1, dom[b]):
                P2 += same(p, np.roll(np.roll(p, s, axis=a), u, axis=b))
pairs = N * (N - 1) / 2
flat = p.ravel()
Edist = float((1 - (1 - flat) ** N).sum())
print(f"échantillon i.i.d. de N = {N} selon P* (espérances) :")
print(f"  distinctes : {Edist:.1f}")
print(f"  paires à distance 0 : {pairs * P0:.0f}, à distance 1 : {pairs * P1:.0f}, à distance ≤ 1 : {pairs * (P0 + P1):.0f}, à distance 2 : {pairs * P2:.0f}")
print(f"  voisins à distance ≤ 1 par individu : {(N - 1) * (P0 + P1):.1f}")
