"""Reproduces Table 9 of Section 5.8 (placement only, replication, recovery).

Run:  python experiments_58.py        (about 10 s; prints the table, writes results_58.json)
"""
import json
from sharing_sim import Params, run_many

SEEDS = range(30)
LOADS = {"light": dict(), "contended": dict(arrival_per_h=15.0, local_busy_min=60.0)}
STRATS = {"P": "Placement only", "R": "Replication (14)-(15)",
          "C": "Recovery (16)-(19), local checkpoints", "Cr": "Recovery (16)-(19), external checkpoints",
          "C+": "Recovery with (19a)-(19b)"}
out = {}
for ld, kw in LOADS.items():
    for s in STRATS:
        r = run_many(Params(**kw), s, SEEDS)
        out[f"{ld}|{s}"] = {k: [float(m), float(c)] for k, (m, c) in r.items()}
        d = out[f"{ld}|{s}"]
        print(f"{ld:9s} {STRATS[s]:42s} success {100*d['dsr'][0]:5.1f} ±{100*d['dsr'][1]:.1f}  "
              f"GPU-h/done {d['energy_kwh_per_done'][0]/Params().power_kw:5.3f}  useful {100*d['useful_ratio'][0]:4.1f}%  "
              f"local-delayed {100*d['local_miss'][0]:4.1f}%  migr/task {d['migrations_per_task'][0]:.2f}")
json.dump(out, open("results_58.json", "w"), indent=1)
