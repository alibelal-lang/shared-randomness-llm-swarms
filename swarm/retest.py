"""Amendment 3, A: test-retest of an Adaption policy (fresh vs original samples of the same prompts).

  python -m swarm.retest policies/adaption_glm-5.2.json policies/adaption_v2/glm-5.2_retest.json

Per prompt, z = (p_fresh - p_orig) / pooled standard error, using J/K counts. If sum z^2 over prompts is below the
chi-square 99th percentile, the samples are judged consistent and may be pooled.
"""

import json
import math
import sys

import numpy as np

CHI2_99 = {84: 117.6}


def counts(path):
    out = {}
    for r in json.load(open(path))["records"]:
        if "error" in r or "counts" not in r:
            continue
        la = r["label_A"]
        lb = "K" if la == "J" else "J"
        out[(r["framing"], r["mapping"], r["history"])] = (r["counts"][la], r["counts"][lb])
    return out


def retest(orig: str, fresh: str) -> dict:
    a, b = counts(orig), counts(fresh)
    keys = sorted(set(a) & set(b))
    zs, worst = [], []
    for k in keys:
        (a1, b1), (a2, b2) = a[k], b[k]
        n1, n2 = a1 + b1, a2 + b2
        p1, p2 = a1 / n1, a2 / n2
        pp = (a1 + a2) / (n1 + n2)
        se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
        z = 0.0 if se == 0 else (p2 - p1) / se
        zs.append(z)
        worst.append((abs(z), k, round(p1, 3), round(p2, 3)))
    zs = np.array(zs)
    stat = float((zs ** 2).sum())
    crit = CHI2_99.get(len(keys), len(keys) + 2.326 * math.sqrt(2 * len(keys)) + 2.0)
    res = {"orig": orig, "fresh": fresh, "prompts": len(keys), "sum_z2": round(stat, 1), "critical_99": crit,
           "consistent": stat < crit, "abs_z_gt_2": int((np.abs(zs) > 2).sum()),
           "mean_abs_diff": round(float(np.mean([abs(w[3] - w[2]) for w in worst])), 4),
           "largest": [{"prompt": f"f{k[0]}m{k[1]} {k[2]}", "z": round(z, 2), "orig": p1, "fresh": p2}
                       for z, k, p1, p2 in sorted(worst, reverse=True)[:5]]}
    return res


if __name__ == "__main__":
    r = retest(sys.argv[1], sys.argv[2])
    print(json.dumps(r, indent=1))
