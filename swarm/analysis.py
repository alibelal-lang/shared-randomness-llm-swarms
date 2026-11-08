"""Layer-2 uncertainty: parametric bootstrap over the sampled (Adaption) policy.

Each replicate draws every history's p_A from Beta(k_A + 0.5, k_B + 0.5) independently for each prompt variant,
reruns fresh population seeds, and recomputes the contrasts. Percentiles over replicates combine policy-estimation
uncertainty with the run-to-run noise of the replicate's seeds.
"""

import json
from collections import defaultdict
from multiprocessing import get_context

import numpy as np

from .game import HistorySpace
from .samplers import parse_sampler
from .simulate import run_population

VARIANTS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def policy_counts(path: str, framing: int, mapping: int):
    """(k_A, k_B) per history code from a sampled policy file."""
    data = json.load(open(path))
    space = HistorySpace(data["meta"]["H"])
    kA, kB = np.zeros(len(space)), np.zeros(len(space))
    for r in data["records"]:
        if r["framing"] != framing or r["mapping"] != mapping or "error" in r:
            continue
        code = space.keys().index(r["history"])
        label_B = "K" if r["label_A"] == "J" else "J"
        kA[code] += r["counts"][r["label_A"]]
        kB[code] += r["counts"][label_B]
    return kA, kB, space


def _cell(args):
    pA, H, N, T, spec, init, seeds, game = args
    space = HistorySpace(H)
    smp = parse_sampler(spec)
    rs = [run_population(pA, space, N, T, smp, init, s, game=game) for s in seeds]
    return float(np.mean([r.mean_x_late for r in rs])), float(np.mean([r.mean_reward for r in rs]))


def bootstrap(path: str, B: int = 100, seeds_per: int = 100, Ns=(4, 32), samplers=("independent", "pair_shared:restore"),
              T: int = 100, init: str = "incumbent_B", variants=VARIANTS, seed: int = 0, processes: int = 4):
    rng = np.random.default_rng(seed)
    counts = {v: policy_counts(path, *v) for v in variants}
    H = next(iter(counts.values()))[2].H
    game = json.load(open(path))["meta"].get("game", "coord")
    jobs, keys = [], []
    for b in range(B):
        seeds = list(range(100000 + b * seeds_per, 100000 + (b + 1) * seeds_per))
        for v in variants:
            kA, kB, _ = counts[v]
            pA = rng.beta(kA + 0.5, kB + 0.5)
            for N in Ns:
                for s in samplers:
                    jobs.append((pA, H, N, T, s, init, seeds, game))
                    keys.append((b, v, N, s))
    with get_context("fork").Pool(processes) as pool:
        out = pool.map(_cell, jobs, chunksize=4)
    x = defaultdict(dict)  # x[b][(v, N, s)] = (late x, reward)
    for (b, v, N, s), val in zip(keys, out):
        x[b][(v, N, s)] = val
    return x, {"B": B, "seeds_per": seeds_per, "Ns": list(Ns), "samplers": list(samplers), "T": T, "init": init,
               "variants": [list(v) for v in variants]}


def contrasts_from_bootstrap(x, variants=VARIANTS, Nbig=32, Nsmall=4, treat="pair_shared:restore",
                             base="independent", idx=0):
    """Per replicate: primary contrast (variant-averaged), per-variant contrasts, interaction, N effect."""
    rows = []
    for b, cells in x.items():
        per_v = {v: cells[(v, Nbig, treat)][idx] - cells[(v, Nbig, base)][idx] for v in variants}
        small = {v: cells[(v, Nsmall, treat)][idx] - cells[(v, Nsmall, base)][idx] for v in variants}
        n_eff = {v: cells[(v, Nbig, base)][idx] - cells[(v, Nsmall, base)][idx] for v in variants}
        rows.append({
            "primary": float(np.mean(list(per_v.values()))),
            "interaction": float(np.mean([per_v[v] - small[v] for v in variants])),
            "n_effect_indep": float(np.mean(list(n_eff.values()))),
            **{f"contrast_f{v[0]}m{v[1]}": per_v[v] for v in variants},
            **{f"n_effect_f{v[0]}m{v[1]}": n_eff[v] for v in variants},
        })
    keys = rows[0].keys()
    return {k: {"mean": float(np.mean([r[k] for r in rows])),
                "lo": float(np.percentile([r[k] for r in rows], 2.5)),
                "hi": float(np.percentile([r[k] for r in rows], 97.5))} for k in keys}
