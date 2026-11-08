"""Summaries of Amendment 3 confirmation sets (results/ext/<set>/<variant>_incumbent_B/runs.csv, bootstrap.json).

  python -m swarm.ext_report results/ext/orig_qwen3.8-27b
"""

import csv
import glob
import json
import math
import os
import sys

import numpy as np

NS = (4, 8, 16, 32, 64)
ALPHAS = ("0.1", "0.2", "0.3", "0.5", "0.75", "1")


def _runs(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out.setdefault((int(r["N"]), r["sampler"]), {})[int(r["seed"])] = (float(r["mean_x_late"]), float(r["mean_reward"]))
    return out


def _paired(runs, N, a, b, i=0):
    seeds = sorted(set(runs[(N, a)]) & set(runs[(N, b)]))
    d = np.array([runs[(N, a)][s][i] - runs[(N, b)][s][i] for s in seeds])
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))


def _avg(items):
    m = float(np.mean([x[0] for x in items]))
    se = float(math.sqrt(sum(x[1] ** 2 for x in items)) / len(items))
    return {"mean": m, "se": se, "lo": m - 1.96 * se, "hi": m + 1.96 * se}


def summarize_set(d: str) -> dict:
    vdirs = sorted(glob.glob(f"{d}/*_incumbent_B"))
    runs = {os.path.basename(v).split("_")[0]: _runs(f"{v}/runs.csv") for v in vdirs}
    out = {"set": os.path.basename(d), "variants": list(runs), "by_N": [], "dose": [], "per_variant": []}
    for N in NS:
        if not all((N, "independent") in r for r in runs.values()):
            continue
        x_ind = float(np.mean([np.mean([v[0] for v in r[(N, "independent")].values()]) for r in runs.values()]))
        x_res = float(np.mean([np.mean([v[0] for v in r[(N, "pair_shared:restore")].values()]) for r in runs.values()]))
        c = _avg([_paired(r, N, "pair_shared:restore", "independent") for r in runs.values()])
        out["by_N"].append({"N": N, "x_independent": x_ind, "x_coupled": x_res, "coupling": c})
    for N in (32, 64):
        for a in ALPHAS + ("restore",):
            s = f"pair_shared:{a}"
            if all((N, s) in r for r in runs.values()):
                out["dose"].append({"N": N, "alpha": a, "coupling": _avg([_paired(r, N, s, "independent") for r in runs.values()])})
    for v, r in runs.items():
        x4 = np.array(list(r[(4, "independent")].values()))
        row = {"variant": v}
        for N in (32, 64):
            xN = np.array(list(r[(N, "independent")].values()))
            row[f"size{N}"] = float(xN[:, 0].mean() - x4[:, 0].mean())
            row[f"coupling{N}"] = _paired(r, N, "pair_shared:restore", "independent")[0]
            row[f"block{N}"] = _paired(r, N, "block:2", "independent")[0]
        out["per_variant"].append(row)
    bp = f"{d}/bootstrap.json"
    if os.path.exists(bp):
        b = json.load(open(bp))
        out["bootstrap"] = {k: b["late_x"][k] for k in ("primary", "interaction", "n_effect_indep")}
        out["bootstrap_reward"] = b["reward"]["primary"]
        out["bootstrap_variants"] = {k: v for k, v in b["late_x"].items() if k.startswith("contrast_")}
    return out


def show(s):
    print(f"== {s['set']} ({', '.join(s['variants'])})")
    for r in s["by_N"]:
        c = r["coupling"]
        print(f"  N={r['N']:>2}  independent {r['x_independent']:.3f}  coupled {r['x_coupled']:.3f}  "
              f"coupling {100 * c['mean']:+5.1f} [{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}]")
    for N in (32, 64):
        ds = [r for r in s["dose"] if r["N"] == N]
        print(f"  dose N={N}: " + "  ".join(f"a={r['alpha']}:{100 * r['coupling']['mean']:+.1f}" for r in ds))
    for r in s["per_variant"]:
        print(f"  {r['variant']}: size32 {100 * r['size32']:+.1f} size64 {100 * r['size64']:+.1f} | coupling32 "
              f"{100 * r['coupling32']:+.1f} coupling64 {100 * r['coupling64']:+.1f} | block32 {100 * r['block32']:+.1f}")
    if "bootstrap" in s:
        b = s["bootstrap"]
        print("  layer-2: " + "  ".join(f"{k} {100 * v['mean']:+.1f} [{100 * v['lo']:+.1f}, {100 * v['hi']:+.1f}]"
                                         for k, v in b.items()))


if __name__ == "__main__":
    for d in sys.argv[1:] or sorted(glob.glob("results/ext/*/")):
        show(summarize_set(d.rstrip("/")))
