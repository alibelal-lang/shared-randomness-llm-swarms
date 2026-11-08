"""Learning pilot data: reward-filtered SFT rows from frozen-policy rollouts.

For a given sampler, run populations with the base model's extracted policy and keep each decision whose payoff
beat that round's population mean payoff (a positive advantage against a round baseline). The sampler changes
which trajectories are visited and therefore which decisions count as good. Rows are subsampled to a fixed
count per prompt variant, so the two arms get equal training budgets.

This is offline reward-filtered fine-tuning, not on-policy RL.
"""

import csv

import numpy as np

from .extract import load_policy
from .game import HistorySpace, initial_codes, payoff
from .prompts import LABEL_MAPPINGS, build_prompt
from .samplers import parse_sampler

VARIANTS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def rollout_decisions(pA, space, N, T, sampler, init, seed):
    match_rng = np.random.default_rng([seed, 0])
    sample_rng = np.random.default_rng([seed, 1])
    init_rng = np.random.default_rng([seed, 2])
    codes = initial_codes(space, init, N, init_rng)
    partner = np.empty(N, dtype=np.int64)
    out = []  # (history code, action, payoff, round mean payoff, round)
    for t in range(T):
        pairs = match_rng.permutation(N).reshape(-1, 2)
        partner[pairs[:, 0]], partner[pairs[:, 1]] = pairs[:, 1], pairs[:, 0]
        a, _ = sampler(pA[codes], pairs, sample_rng)
        ap = a[partner]
        pay = payoff(a, ap)
        mean_pay = pay.mean()
        out.extend(zip(codes.tolist(), a.tolist(), pay.tolist(), [mean_pay] * N, [t + 1] * N))
        codes = space.next[codes, a, ap]
    return out


def build_sft_rows(policy_path, sampler_spec, N=32, T=100, seeds=range(200000, 200200), init="incumbent_B",
                   rows_per_variant=1000, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    sampler = parse_sampler(sampler_spec)
    rows, stats = [], []
    for f, m in VARIANTS:
        pA, space, _ = load_policy(policy_path, f, m)
        kept = []
        total = 0
        for s in seeds:
            for code, act, pay, mean_pay, t in rollout_decisions(pA, space, N, T, sampler, init, s):
                total += 1
                if pay > mean_pay:
                    kept.append((code, act))
        if len(kept) < rows_per_variant:
            raise ValueError(f"f{f}m{m}: only {len(kept)} positive-advantage decisions; lower rows_per_variant")
        pick = rng.choice(len(kept), size=rows_per_variant, replace=False)
        lab = LABEL_MAPPINGS[m]
        chosen = [kept[i] for i in pick]
        for code, act in chosen:
            rows.append({"prompt": build_prompt(space.histories[code], f, m), "completion": lab[act],
                         "history": space.key(space.histories[code]), "action": "A" if act == 1 else "B",
                         "framing": f, "mapping": m})
        stats.append({"framing": f, "mapping": m, "decisions": total, "kept": len(kept),
                      "kept_frac": len(kept) / total, "share_A_in_rows": float(np.mean([a for _, a in chosen])),
                      "distinct_histories": len({c for c, _ in chosen})})
    return rows, stats


def write_rows(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
