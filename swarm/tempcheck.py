"""Amendment 3, D3: is Adaption's temperature the textbook rescaling of the same logits?

Compares P(A) sampled at temperature 0.7 with sigmoid(logit(p_T1) / 0.7) predicted from the temperature-1 sample of
the same deployment (the fresh retest). Adaption also applies top_p = 0.95, which truncates small probabilities
differently at each temperature, so a gap is expected near 0 and 1.
"""

import json
import sys

import numpy as np

from .extract import load_policy

VARIANTS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def check(t1: str, t07: str, T: float = 0.7) -> dict:
    pred, obs = [], []
    for v in VARIANTS:
        pred.append(load_policy(t1, *v, temperature=T)[0])
        obs.append(load_policy(t07, *v)[0])
    pred, obs = np.concatenate(pred), np.concatenate(obs)
    n = 200
    z = (obs - pred) / np.sqrt(np.clip(pred * (1 - pred), 1e-4, None) / n)
    mid = (pred > 0.05) & (pred < 0.95)
    return {"prompts": int(len(obs)), "corr": round(float(np.corrcoef(obs, pred)[0, 1]), 3),
            "mean_abs_diff": round(float(np.abs(obs - pred).mean()), 4),
            "mean_diff": round(float((obs - pred).mean()), 4),
            "abs_z_gt_2": int((np.abs(z) > 2).sum()),
            "mid_prompts": int(mid.sum()), "mid_mean_abs_diff": round(float(np.abs(obs - pred)[mid].mean()), 4),
            "more_extreme_than_predicted": round(float(np.mean(np.abs(obs - 0.5) > np.abs(pred - 0.5))), 3)}


if __name__ == "__main__":
    r = check(sys.argv[1], sys.argv[2])
    json.dump(r, open("results/ext/tempcheck_qwen3.8-27b.json", "w"), indent=1)
    print(json.dumps(r, indent=1))
