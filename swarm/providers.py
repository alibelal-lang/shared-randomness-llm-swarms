"""Amendment 3, step 1: the scoping prompt set replicated on several OpenRouter providers per model.

For every provider file: label-mass validity, agreement with the Adaption policy, and the simulated
coupling and size effects (independent vs pair_shared:restore, N = 4 / 32, seed-paired). Writes
results/providers/summary.csv and prints the pinned provider per model (validity gate, then smallest
mean |p - p_Adaption|).
"""

import csv
import glob
import json
import os
from multiprocessing import get_context

import numpy as np

from .extract import load_policy
from .game import HistorySpace
from .samplers import parse_sampler
from .simulate import run_population

VARIANTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
SEEDS = range(300)
CELLS = [(4, "independent"), (32, "independent"), (32, "pair_shared:restore")]


def _cell(args):
    pA, H, N, spec, game = args
    space, smp = HistorySpace(H), parse_sampler(spec)
    return float(np.mean([run_population(pA, space, N, 100, smp, "incumbent_B", s, game=game).mean_x_late
                          for s in SEEDS]))


def effects(path: str, pool) -> dict:
    meta = json.load(open(path))["meta"]
    out = {}
    jobs, keys = [], []
    for v in VARIANTS:
        pA, space, _ = load_policy(path, *v)
        for N, s in CELLS:
            jobs.append((pA, space.H, N, s, meta.get("game", "coord")))
            keys.append((v, N, s))
    res = dict(zip(keys, pool.map(_cell, jobs)))
    for v in VARIANTS:
        out[v] = {"coupling": res[(v, 32, "pair_shared:restore")] - res[(v, 32, "independent")],
                  "size": res[(v, 32, "independent")] - res[(v, 4, "independent")]}
    return out


def main():
    rows = []
    with get_context("fork").Pool(8) as pool:
        for path in sorted(glob.glob("policies/openrouter/providers/*.json")):
            short, prov = os.path.basename(path)[:-5].split("__")
            data = json.load(open(path))
            meta = data["meta"]
            ad = f"policies/adaption_{short}.json"
            pv, pa_all, val = [], [], []
            for v in VARIANTS:
                p, _, d = load_policy(path, *v)
                pa, _, _ = load_policy(ad, *v)
                pv.append(p)
                pa_all.append(pa)
                val.append(d["mean_valid_mass"])
            pv, pa_all = np.concatenate(pv), np.concatenate(pa_all)
            eff = effects(path, pool)
            row = {"model": short, "provider": prov, "errors": meta["errors"], "cost_usd": round(meta["total_cost"], 5),
                   "valid_mass": round(float(np.mean(val)), 4), "min_valid_mass": round(float(np.min(val)), 4),
                   "mad_vs_adaption": round(float(np.abs(pv - pa_all).mean()), 4),
                   "corr_vs_adaption": round(float(np.corrcoef(pv, pa_all)[0, 1]), 3) if pv.std() > 0 and pa_all.std() > 0 else None,
                   "coupling_pp": round(100 * float(np.mean([e["coupling"] for e in eff.values()])), 2),
                   "size_pp": round(100 * float(np.mean([e["size"] for e in eff.values()])), 2)}
            for v, e in eff.items():
                row[f"coupling_f{v[0]}m{v[1]}"] = round(100 * e["coupling"], 2)
                row[f"size_f{v[0]}m{v[1]}"] = round(100 * e["size"], 2)
            rows.append(row)
            print({k: row[k] for k in ("model", "provider", "valid_mass", "mad_vs_adaption", "corr_vs_adaption",
                                       "coupling_pp", "size_pp", "cost_usd")}, flush=True)
    os.makedirs("results/providers", exist_ok=True)
    with open("results/providers/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    pins = {}
    for m in sorted({r["model"] for r in rows}):
        ok = [r for r in rows if r["model"] == m and r["errors"] == 0 and r["min_valid_mass"] >= 0.9]
        if ok:
            pins[m] = min(ok, key=lambda r: r["mad_vs_adaption"])["provider"]
    json.dump(pins, open("results/providers/pins.json", "w"), indent=1)
    print("pinned:", pins)


if __name__ == "__main__":
    main()
