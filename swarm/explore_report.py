"""Summaries of the exploration grid (results/explore/grid.csv) and the Amendment 3 D flagging rule."""

import csv
import json
from collections import defaultdict

import numpy as np

from .prompts import FIRST_LETTER, FRAMING_NAMES, LABEL_MAPPINGS
from .game import A

G = "results/explore/grid.csv"
NUM = ("p_incumbent", "p_AA", "mean_p_full", "size32_pp", "size64_pp", "coupling32_pp", "coupling64_pp",
       "size32_se_pp", "coupling32_se_pp", "x_ind4", "x_ind32", "x_ind64", "x_res32", "x_res64", "reward_gain32",
       "min_valid_mass", "temperature")


def load():
    rows = list(csv.DictReader(open(G)))
    for r in rows:
        for k in NUM:
            if k in r and r[k] != "":
                r[k] = float(r[k])
        for k in ("H", "framing", "mapping"):
            r[k] = int(r[k])
        r["pinned"] = r["pinned"] == "True"
        r["responsive"] = r["responsive"] == "True"
        r["a_letter"] = LABEL_MAPPINGS[r["mapping"]][A]
        r["a_first"] = FIRST_LETTER[r["mapping"]] == r["a_letter"]
    return rows


def flag(rows, model, already=()):
    """Amendment 3 D2: up to 6 responsive H=2, T=1 variants on the pinned deployment, most negative size first."""
    c = [r for r in rows if r["model"] == model and r["pinned"] and r["H"] == 2 and r["temperature"] == 1.0
         and r["responsive"] and (r["game"], r["framing"], r["mapping"]) not in already]
    c.sort(key=lambda r: r["size32_pp"])
    return [(r["game"], r["framing"], r["mapping"], r["size32_pp"], r["coupling32_pp"]) for r in c[:6]]


def main():
    rows = load()
    base = [r for r in rows if r["H"] == 2 and r["temperature"] == 1.0]
    print("== Responsive variants (size effect at N=32 <= -2 points), H=2, T=1, out of 32 per game ==")
    tab = defaultdict(lambda: [0, 0])
    for r in base:
        k = (r["model"], r["deployment"] + ("*" if r["pinned"] else ""), r["game"])
        tab[k][0] += r["responsive"]
        tab[k][1] += 1
    for k in sorted(tab):
        print(f"  {k[0]:<20} {k[1]:<16} {k[2]:<6} {tab[k][0]:>2}/{tab[k][1]}")

    print("\n== Mechanism: coupling vs size effect at N=32, all H=2 T=1 variants ==")
    s = np.array([r["size32_pp"] for r in base])
    c = np.array([r["coupling32_pp"] for r in base])
    resp = s <= -2
    print(f"  variants {len(s)}, responsive {resp.sum()}; corr(coupling, size) {np.corrcoef(c, s)[0, 1]:.2f}")
    if resp.any():
        rec = -c[resp] / s[resp]
        print(f"  responsive: coupling > 0 in {(c[resp] > 0).sum()}/{resp.sum()}, median share of size penalty recovered "
              f"{np.median(rec):.0%} (IQR {np.percentile(rec, 25):.0%}-{np.percentile(rec, 75):.0%})")
    print(f"  non-responsive: mean |coupling| {np.abs(c[~resp]).mean():.2f} points")

    print("\n== Scale: N=64 vs N=32 among responsive variants ==")
    rr = [r for r in base if r["responsive"]]
    if rr:
        print(f"  size effect mean {np.mean([r['size32_pp'] for r in rr]):+.1f} (N=32) vs "
              f"{np.mean([r['size64_pp'] for r in rr]):+.1f} (N=64); coupling {np.mean([r['coupling32_pp'] for r in rr]):+.1f}"
              f" vs {np.mean([r['coupling64_pp'] for r in rr]):+.1f}; larger penalty at 64 in "
              f"{sum(r['size64_pp'] < r['size32_pp'] for r in rr)}/{len(rr)}")

    print("\n== Game: responsive share and mean coupling, pinned deployments ==")
    for g in ("coord", "stag"):
        gg = [r for r in base if r["pinned"] and r["game"] == g]
        print(f"  {g:<6} responsive {sum(r['responsive'] for r in gg)}/{len(gg)}, mean coupling "
              f"{np.mean([r['coupling32_pp'] for r in gg]):+.2f}, mean size {np.mean([r['size32_pp'] for r in gg]):+.2f}, "
              f"mean P(A | incumbent B) {np.mean([r['p_incumbent'] for r in gg]):.2f}")

    print("\n== Labels: mean P(A | incumbent B history) by letter and list order, pinned, H=2, T=1 ==")
    for m in sorted({r["model"] for r in base}):
        for g in ("coord", "stag"):
            gg = [r for r in base if r["pinned"] and r["model"] == m and r["game"] == g]
            if not gg:
                continue
            f = lambda pred: np.mean([r["p_incumbent"] for r in gg if pred(r)])  # noqa: E731
            print(f"  {m:<20} {g:<6} A=J {f(lambda r: r['a_letter'] == 'J'):.2f} vs A=K {f(lambda r: r['a_letter'] == 'K'):.2f}"
                  f" | A listed first {f(lambda r: r['a_first']):.2f} vs second {f(lambda r: not r['a_first']):.2f}")

    print("\n== Wording: responsive variants per framing (pinned, H=2, T=1, both games, all models) ==")
    for fr in range(8):
        gg = [r for r in base if r["pinned"] and r["framing"] == fr]
        print(f"  f{fr} {FRAMING_NAMES[fr]:<24} {sum(r['responsive'] for r in gg):>2}/{len(gg)}")

    print("\n== Memory: responsive variants by H (pinned, framings 0-1, T=1) ==")
    for H in (1, 2, 3):
        gg = [r for r in rows if r["pinned"] and r["H"] == H and r["framing"] in (0, 1) and r["temperature"] == 1.0]
        by = defaultdict(int)
        for r in gg:
            by[(r["model"], r["game"])] += r["responsive"]
        print(f"  H={H}: {sum(r['responsive'] for r in gg)}/{len(gg)}  " +
              ", ".join(f"{m.split('-')[0]}{'/' + g}={n}" for (m, g), n in sorted(by.items()) if n))

    print("\n== Temperature: responsive variants by T (pinned, H=2) ==")
    for T in (0.5, 0.75, 1.0, 1.5):
        gg = [r for r in rows if r["pinned"] and r["H"] == 2 and r["temperature"] == T]
        by = defaultdict(int)
        for r in gg:
            by[r["model"]] += r["responsive"]
        print(f"  T={T}: {sum(r['responsive'] for r in gg)}/{len(gg)}  " + ", ".join(f"{m}={n}" for m, n in sorted(by.items())))

    print("\n== Deployment: agreement of size effects across providers (H=2, T=1, same 64 variants) ==")
    for m in sorted({r["model"] for r in base}):
        deps = sorted({r["deployment"] for r in base if r["model"] == m})
        if len(deps) < 2:
            continue
        vec = {d: {(r["game"], r["framing"], r["mapping"]): r["size32_pp"] for r in base if r["model"] == m and r["deployment"] == d}
               for d in deps}
        keys = sorted(set.intersection(*[set(v) for v in vec.values()]))
        resp = {d: sum(vec[d][k] <= -2 for k in keys) for d in deps}
        print(f"  {m}: responsive per deployment {resp}")

    already = {("stag", f, m) for f in (0, 1) for m in (0, 1)} | {("coord", f, m) for f in (0, 1) for m in (0, 1)}
    flags = {m: flag(rows, m, already) for m in ("qwen3.8-27b", "glm-5.2")}
    print("\n== Amendment 3 D2 flags (game, framing, mapping, size32, coupling32) ==")
    for m, fl in flags.items():
        print(f"  {m}: {fl}")
    json.dump(flags, open("results/explore/flags.json", "w"), indent=1)


if __name__ == "__main__":
    main()
