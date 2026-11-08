"""Amendment 3, C: simulate every explored prompt variant and record size and coupling effects.

  python -m swarm.sweep explore            # all policies/openrouter/grid/*.json -> results/explore/parts/*.csv
  python -m swarm.sweep collect            # parts -> results/explore/grid.csv

Cells per variant: independent at N = 4, 32, 64; pair_shared:restore at N = 32, 64. Seeds 0-199, T = 100,
incumbent_B start. Temperatures 0.5, 0.75, 1, 1.5 on pinned deployments at H = 2, otherwise T = 1 only.
"""

import csv
import glob
import json
import math
import os
import sys
from multiprocessing import get_context

import numpy as np

from .extract import load_policy
from .game import HistorySpace
from .samplers import parse_sampler
from .simulate import run_population

SEEDS = range(200)
CELLS = [(4, "independent"), (32, "independent"), (64, "independent"),
         (32, "pair_shared:restore"), (64, "pair_shared:restore")]
TEMPS = (0.5, 0.75, 1.0, 1.5)
OUT = "results/explore"


def _cell(args):
    key, pA, H, N, spec, game = args
    space, smp = HistorySpace(H), parse_sampler(spec)
    rs = [run_population(pA, space, N, 100, smp, "incumbent_B", s, game=game) for s in SEEDS]
    return key, np.array([r.mean_x_late for r in rs]), np.array([r.mean_reward for r in rs])


def _pins():
    return {m: p for m, p in json.load(open("results/providers/pins.json")).items()}


def _rows_for(path, pool):
    meta = json.load(open(path))["meta"]
    short, prov, game, Htag = os.path.basename(path)[:-5].split("__")
    pinned = _pins().get(short) == prov
    temps = TEMPS if (pinned and meta["H"] == 2) else (1.0,)
    variants = [(f, m) for f in meta["framings"] for m in meta["mappings"]]
    jobs, info = [], {}
    for T in temps:
        for f, m in variants:
            pA, space, d = load_policy(path, f, m, temperature=T)
            inc = space.keys().index(".".join(["BB"] * space.H))
            aa = space.keys().index(".".join(["AA"] * space.H))
            full = [i for i, k in enumerate(space.keys()) if k.count(".") == space.H - 1 and len(k) == 3 * space.H - 1]
            info[(T, f, m)] = {"p_incumbent": float(pA[inc]), "p_AA": float(pA[aa]),
                               "mean_p_full": float(pA[full].mean()),
                               "intermediate": int(sum(0.05 < pA[i] < 0.95 for i in full)),
                               "min_valid_mass": d["min_valid_mass"], "bounded": len(d["bounded_histories"])}
            for N, s in CELLS:
                jobs.append(((T, f, m, N, s), pA, space.H, N, s, game))
    res = {}
    for key, x, r in pool.imap_unordered(_cell, jobs, chunksize=2):
        res[key] = (x, r)
    rows = []
    n = len(SEEDS)
    for (T, f, m), inf in info.items():
        g = lambda N, s, i=0: res[(T, f, m, N, s)][i]  # noqa: E731
        row = {"model": short, "deployment": prov, "pinned": pinned, "game": game, "H": meta["H"], "framing": f,
               "mapping": m, "temperature": T, **{k: round(v, 4) if isinstance(v, float) else v for k, v in inf.items()}}
        for N, s in CELLS:
            row[f"x_{'ind' if s == 'independent' else 'res'}{N}"] = round(float(g(N, s).mean()), 4)
        for N in (32, 64):
            size = g(N, "independent").mean() - g(4, "independent").mean()
            size_se = math.sqrt(g(N, "independent").var(ddof=1) / n + g(4, "independent").var(ddof=1) / n)
            d = g(N, "pair_shared:restore") - g(N, "independent")
            dr = g(N, "pair_shared:restore", 1) - g(N, "independent", 1)
            row.update({f"size{N}_pp": round(100 * size, 2), f"size{N}_se_pp": round(100 * size_se, 2),
                        f"coupling{N}_pp": round(100 * d.mean(), 2),
                        f"coupling{N}_se_pp": round(100 * d.std(ddof=1) / math.sqrt(n), 2),
                        f"reward_gain{N}": round(float(dr.mean()), 4)})
        row["responsive"] = row["size32_pp"] <= -2.0
        rows.append(row)
    return rows


def explore(paths=None):
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    paths = paths or sorted(glob.glob("policies/openrouter/grid/*.json"))
    with get_context("fork").Pool(max(1, (os.cpu_count() or 4) - 1)) as pool:
        for p in paths:
            part = f"{OUT}/parts/{os.path.basename(p)[:-5]}.csv"
            if os.path.exists(part) and os.path.getmtime(part) > os.path.getmtime(p):
                continue
            if json.load(open(p))["meta"]["errors"]:
                print(f"SKIP {os.path.basename(p)}: has failed prompts; re-extract it first", flush=True)
                continue
            rows = _rows_for(p, pool)
            with open(part, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            nr = sum(r["responsive"] for r in rows if r["temperature"] == 1.0)
            print(f"{os.path.basename(p)}: {len(rows)} rows, responsive at T=1: {nr}", flush=True)


def collect():
    rows = []
    for p in sorted(glob.glob(f"{OUT}/parts/*.csv")):
        rows += list(csv.DictReader(open(p)))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(f"{OUT}/grid.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{OUT}/grid.csv: {len(rows)} rows")
    return rows


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "explore"
    if cmd == "explore":
        explore(sys.argv[2:] or None)
    collect()
