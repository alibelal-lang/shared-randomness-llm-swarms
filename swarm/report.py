"""Collect confirmatory results into one JSON per model, following PROTOCOL.md (layer-1, seed-paired intervals)."""

import csv
import json
import math
import os

import numpy as np

VARIANTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
SAMPLERS = ["independent", "pair_shared:restore", "pair_shared:0.3", "block:2"]
NS = [4, 8, 16, 32]


def _runs(path):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for r in rows:
        out[(int(r["N"]), r["sampler"], int(r["seed"]))] = r
    return out


def _paired(runs, N, a, b, key):
    seeds = sorted({s for (n, smp, s) in runs if n == N and smp == a})
    d = np.array([float(runs[(N, a, s)][key]) - float(runs[(N, b, s)][key]) for s in seeds])
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d))), len(d)


def summarize_model(results_dir: str, init: str = "incumbent_B") -> dict:
    cells, contrasts = {}, {}
    for f, m in VARIANTS:
        v = f"f{f}m{m}"
        d = os.path.join(results_dir, f"{v}_{init}")
        if not os.path.exists(os.path.join(d, "runs.csv")):
            continue
        summ = {(int(r["N"]), r["sampler"]): r for r in csv.DictReader(open(os.path.join(d, "summary.csv")))}
        cells[v] = {f"N{N}|{s}": {k: float(summ[(N, s)][k]) for k in
                                  ("mean_x_late", "mean_reward", "entry_prob", "final_in_prob", "all_A_prob",
                                   "occupancy", "mean_alpha", "infeasible_frac")}
                    for N in NS for s in SAMPLERS if (N, s) in summ}
        runs = _runs(os.path.join(d, "runs.csv"))
        c = {}
        for key in ("mean_x_late", "mean_reward"):
            c[f"restore_vs_indep_N32|{key}"] = _paired(runs, 32, "pair_shared:restore", "independent", key)
            c[f"restore_vs_indep_N4|{key}"] = _paired(runs, 4, "pair_shared:restore", "independent", key)
            c[f"block_vs_indep_N32|{key}"] = _paired(runs, 32, "block:2", "independent", key)
            c[f"restore_vs_block_N32|{key}"] = _paired(runs, 32, "pair_shared:restore", "block:2", key)
            ind32 = np.array([float(runs[(32, "independent", s)][key]) for s in range(500) if (32, "independent", s) in runs])
            ind4 = np.array([float(runs[(4, "independent", s)][key]) for s in range(500) if (4, "independent", s) in runs])
            c[f"n_effect_indep|{key}"] = (float(ind32.mean() - ind4.mean()),
                                          float(math.sqrt(ind32.var(ddof=1) / len(ind32) + ind4.var(ddof=1) / len(ind4))),
                                          len(ind32))
        # entry at x >= 0.9 is not stored per run, so it is approximated from final_in only when needed (secondary)
        contrasts[v] = {k: {"mean": m_, "se": se, "lo": m_ - 1.96 * se, "hi": m_ + 1.96 * se, "runs": n}
                        for k, (m_, se, n) in c.items()}
    avg = {}
    if contrasts:
        for k in next(iter(contrasts.values())):
            ms = [contrasts[v][k]["mean"] for v in contrasts]
            ses = [contrasts[v][k]["se"] for v in contrasts]
            m_ = float(np.mean(ms))
            se = float(math.sqrt(sum(s * s for s in ses)) / len(ses))
            avg[k] = {"mean": m_, "se": se, "lo": m_ - 1.96 * se, "hi": m_ + 1.96 * se, "variants": len(ms)}
    return {"results_dir": results_dir, "init": init, "cells": cells, "contrasts": contrasts, "variant_average": avg}


def print_summary(s: dict):
    print(f"{s['results_dir']} ({s['init']})")
    print(f"  {'variant':<8}" + "".join(f"{'N=' + str(N) + ' ind / restore / block':>30}" for N in (4, 32)))
    for v, cells in s["cells"].items():
        line = f"  {v:<8}"
        for N in (4, 32):
            g = lambda smp: cells.get(f"N{N}|{smp}", {}).get("mean_x_late", float("nan"))
            line += f"{g('independent'):>14.3f} / {g('pair_shared:restore'):.3f} / {g('block:2'):.3f}"
        print(line)
    print("  Contrasts on late-window share of A (seed-paired 95% CI):")
    for v, c in list(s["contrasts"].items()) + [("AVERAGE", s["variant_average"])]:
        a = c["restore_vs_indep_N32|mean_x_late"]
        n = c["n_effect_indep|mean_x_late"]
        b = c["restore_vs_block_N32|mean_x_late"]
        print(f"    {v:<8} restore-indep@32 {a['mean']:+.3f} [{a['lo']:+.3f},{a['hi']:+.3f}]   "
              f"N-effect(indep 32-4) {n['mean']:+.3f} [{n['lo']:+.3f},{n['hi']:+.3f}]   "
              f"restore-block@32 {b['mean']:+.3f} [{b['lo']:+.3f},{b['hi']:+.3f}]")
