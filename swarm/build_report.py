"""Gather every result into small JSON datasets for the findings dashboard (report/data/*.json)."""

import csv
import json
import os

import numpy as np

from .extract import load_policy
from .game import HistorySpace
from .reference import reference_table
from .report import summarize_model

OUT = "report/data"
MODELS = [  # (slug, display name, family, samples per prompt)
    ("qwen3.8-27b", "Qwen 3.8 27B", "Qwen", 200),
    ("qwen3.5-397b", "Qwen 3.5 397B", "Qwen", 100),
    ("gemma-4-31b-it", "Gemma 4 31B", "Gemma", 100),
    ("gemma-4-26b-a4b-it", "Gemma 4 26B A4B", "Gemma", 100),
    ("glm-5.2", "GLM 5.2", "GLM", 100),
]
VARIANTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
VNAME = {"f0m0": "Framing 1, J better", "f0m1": "Framing 1, K better", "f1m0": "Framing 2, J better",
         "f1m1": "Framing 2, K better"}


def _w(name, rows):
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(f"{OUT}/{name}.json", "w"), indent=0)
    return len(rows)


def build():
    space = HistorySpace(2)
    headline, vcontr, scale, policy, screening = [], [], [], [], []
    for slug, name, fam, n in MODELS:
        pol = f"policies/adaption_{slug}.json"
        if not os.path.exists(pol):
            screening.append({"model": name, "family": fam, "status": "Pending", "invalid_rate": None,
                              "p_BB_BB": None, "p_AA_AA": None, "samples": n})
            continue
        diags, pmap = [], {}
        for f, m in VARIANTS:
            pA, _, d = load_policy(pol, f, m)
            diags.append(d)
            for code, key in enumerate(space.keys()):
                policy.append({"model": name, "variant": f"f{f}m{m}", "history": key, "p_A": round(float(pA[code]), 4)})
            pmap[(f, m)] = pA
        inv = float(np.mean([d["invalid_rate"] for d in diags]))
        bb, aa = space.keys().index("BB.BB"), space.keys().index("AA.AA")
        screening.append({"model": name, "family": fam, "status": "Included" if inv < 0.10 else "Failed gate",
                          "invalid_rate": round(inv, 4), "samples": n,
                          "p_BB_BB": round(float(np.mean([pmap[v][bb] for v in VARIANTS])), 3),
                          "p_AA_AA": round(float(np.mean([pmap[v][aa] for v in VARIANTS])), 3)})
        cdir = f"results/confirm/{slug}"
        if os.path.exists(f"{cdir}/f0m0_incumbent_B/summary.csv"):
            s = summarize_model(cdir, "incumbent_B")
            for v, cells in s["cells"].items():
                for k, val in cells.items():
                    N, smp = k.split("|")
                    scale.append({"model": name, "variant": v, "N": int(N[1:]), "sampler": smp,
                                  "late_x": round(val["mean_x_late"], 4), "reward": round(val["mean_reward"], 4)})
        bpath = f"{cdir}/bootstrap.json"
        if os.path.exists(bpath):
            b = json.load(open(bpath))
            lx, rw = b["late_x"], b["reward"]
            claimed = lx["primary"]["lo"] > 0 or lx["primary"]["hi"] < 0
            pp = lambda x: round(100 * x, 1)  # share differences in percentage points
            headline.append({"model": name, "family": fam,
                             "primary_pp": pp(lx["primary"]["mean"]), "primary_lo_pp": pp(lx["primary"]["lo"]),
                             "primary_hi_pp": pp(lx["primary"]["hi"]),
                             "interaction_pp": pp(lx["interaction"]["mean"]),
                             "interaction_lo_pp": pp(lx["interaction"]["lo"]),
                             "interaction_hi_pp": pp(lx["interaction"]["hi"]),
                             "n_effect_pp": pp(lx["n_effect_indep"]["mean"]),
                             "n_effect_lo_pp": pp(lx["n_effect_indep"]["lo"]),
                             "n_effect_hi_pp": pp(lx["n_effect_indep"]["hi"]),
                             "reward": round(rw["primary"]["mean"], 3), "reward_lo": round(rw["primary"]["lo"], 3),
                             "reward_hi": round(rw["primary"]["hi"], 3),
                             "decision": "effect claimed" if claimed else "no effect claimed"})
            for v in VARIANTS:
                key = f"contrast_f{v[0]}m{v[1]}"
                for metric, block in (("late_x", lx), ("reward", rw)):
                    vcontr.append({"model": name, "variant": f"f{v[0]}m{v[1]}", "variant_name": VNAME[f"f{v[0]}m{v[1]}"],
                                   "metric": metric, "mean": round(block[key]["mean"], 4),
                                   "lo": round(block[key]["lo"], 4), "hi": round(block[key]["hi"], 4)})
    counts = {"headline": _w("headline", headline), "variant_contrasts": _w("variant_contrasts", vcontr),
              "scale": _w("scale", scale), "policy": _w("policy", policy), "screening": _w("screening", screening)}

    # Reference check against the research plan
    plan = {4: (0.3911, 0.3911, 0.3085), 8: (0.0255, 0.6308, 0.4153), 16: (0.00016, 0.6941, 0.4295),
            32: ("< 0.00001", 0.7157, 0.4314)}
    ref = [{"N": r["N"], "indep_first": float(f"{r['independent_first_entry']:.2g}") if r["N"] == 32
            else round(r["independent_first_entry"], 5 if r["N"] == 16 else 4),
            "restore_first": round(r["pair_restore_first_entry"], 4),
            "restore_terminal": round(r["pair_restore_terminal"], 4), "plan_indep_first": plan[r["N"]][0],
            "plan_restore_first": plan[r["N"]][1], "plan_restore_terminal": plan[r["N"]][2]} for r in reference_table()]
    counts["reference"] = _w("reference", ref)

    # Adaption sampled vs OpenRouter exact for the primary model
    agree = []
    exact = "policies/screening/openrouter_qwen3.8-27b_exact.json"
    if os.path.exists(exact) and os.path.exists("policies/adaption_qwen3.8-27b.json"):
        for f, m in VARIANTS:
            pa, _, _ = load_policy("policies/adaption_qwen3.8-27b.json", f, m)
            pb, _, _ = load_policy(exact, f, m)
            z = (pa - pb) / np.sqrt(np.clip(pb * (1 - pb), 1e-4, None) / 200)
            agree.append({"variant": f"f{f}m{m}", "variant_name": VNAME[f"f{f}m{m}"],
                          "corr": round(float(np.corrcoef(pa, pb)[0, 1]), 3),
                          "mean_abs_diff": round(float(np.abs(pa - pb).mean()), 3),
                          "beyond_noise": int((np.abs(z) > 2).sum()), "histories": len(pa)})
    counts["agreement"] = _w("agreement", agree)

    # Live validation
    live, trace = [], []
    ldir = "results/live/qwen3.8-27b"
    if os.path.exists(f"{ldir}/summary.csv"):
        for r in csv.DictReader(open(f"{ldir}/summary.csv")):
            live.append({"variant": f"f{r['framing']}m{r['mapping']}",
                         "variant_name": VNAME[f"f{r['framing']}m{r['mapping']}"], "N": int(r["N"]),
                         "runs": int(r["runs"]), "decisions": int(r["decisions"]),
                         "live_late_x": round(float(r["live_late_x"]), 4), "sim_late_x": round(float(r["sim_late_x_500"]), 4),
                         "sim_ci": round(1.96 * float(r["sim_late_x_500_sd_of_20run_mean"]), 4),
                         "live_mean_action": round(float(r["live_mean_action"]), 4),
                         "extracted_mean_p": round(float(r["extracted_mean_p"]), 4),
                         "calibration_gap": round(float(r["calibration_gap"]), 4),
                         "calibration_ci": round(1.96 * float(r["calibration_gap_se"]), 4),
                         "fallback_rate": round(float(r["fallback_rate"]), 4), "credits": round(float(r["cost"]), 4)})
            v, N = f"f{r['framing']}m{r['mapping']}", int(r["N"])
            for t in csv.DictReader(open(f"{ldir}/trace_{v}_N{N}.csv")):
                trace.append({"variant": v, "N": N, "series": f"{v}|N{N}", "round": int(t["round"]),
                              "live_x": round(float(t["live_x"]), 4), "sim_x": round(float(t["sim_x"]), 4)})
    counts["live"] = _w("live", live)
    counts["live_trace"] = _w("live_trace", trace)

    # Learning pilot
    learn = []
    arms = [("Base model", "policies/local_qwen3.5-4b.json", "results/learning/base_qwen3.5-4b"),
            ("Trained on independent rollouts", "policies/learning/qwen3.5-4b_sft_independent.json",
             "results/learning/sft_independent"),
            ("Trained on coupled rollouts", "policies/learning/qwen3.5-4b_sft_coupled.json",
             "results/learning/sft_coupled")]
    stats = json.load(open("data/learning/build_stats.json")) if os.path.exists("data/learning/build_stats.json") else {}
    for arm, pol, sdir in arms:
        if not os.path.exists(pol):
            continue
        for f, m in VARIANTS:
            pA, _, _ = load_policy(pol, f, m)
            row = {"arm": arm, "variant": f"f{f}m{m}", "variant_name": VNAME[f"f{f}m{m}"],
                   "mean_p_A": round(float(pA.mean()), 4),
                   "p_BB_BB": round(float(pA[space.keys().index("BB.BB")]), 4),
                   "p_AA_AA": round(float(pA[space.keys().index("AA.AA")]), 4)}
            sp = f"{sdir}/f{f}m{m}_incumbent_B/summary.csv"
            if os.path.exists(sp):
                summ = {(int(r["N"]), r["sampler"]): r for r in csv.DictReader(open(sp))}
                row["late_x_N32_indep"] = round(float(summ[(32, "independent")]["mean_x_late"]), 4)
                row["late_x_N32_restore"] = round(float(summ[(32, "pair_shared:restore")]["mean_x_late"]), 4)
            arm_key = {"Trained on independent rollouts": "independent", "Trained on coupled rollouts": "coupled"}.get(arm)
            if arm_key and arm_key in stats:
                st = [s for s in stats[arm_key] if s["framing"] == f and s["mapping"] == m][0]
                row["train_share_A"] = round(st["share_A_in_rows"], 4)
            learn.append(row)
    counts["learning"] = _w("learning", learn)

    # Adaption usage
    usage = []
    for slug, name, fam, n in MODELS:
        p = f"policies/adaption_{slug}.json"
        if os.path.exists(p):
            meta = json.load(open(p))["meta"]
            usage.append({"item": f"Policy extraction: {name}", "calls": meta["calls"], "credits": round(meta["total_cost"], 4)})
    if live:
        usage.append({"item": "Live validation: Qwen 3.8 27B", "calls": sum(r["decisions"] for r in live),
                      "credits": round(sum(r["credits"] for r in live), 4)})
    runs_p = "data/learning/autoscientist_runs.json"
    if os.path.exists(runs_p):
        for arm in ("independent", "coupled"):
            rp = f"data/learning/run_{arm}.json"
            if os.path.exists(rp):
                usage.append({"item": f"AutoScientist fine-tune: {arm} rollouts", "calls": 1, "credits": None})
    counts["usage"] = _w("usage", usage)
    return counts


if __name__ == "__main__":
    print(build())
