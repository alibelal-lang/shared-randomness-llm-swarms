"""Amendment 3 datasets for the findings dashboard (report/data/ext_*.json)."""

import csv
import glob
import json
import os

import numpy as np

from .explore_report import load as load_grid
from .ext_report import summarize_set
from .prompts import FRAMING_NAMES

OUT = "report/data"
NAMES = {"qwen3.8-27b": "Qwen 3.8 27B", "glm-5.2": "GLM 5.2", "qwen3.5-397b": "Qwen 3.5 397B",
         "gemma-4-31b-it": "Gemma 4 31B", "gemma-4-26b-a4b-it": "Gemma 4 26B A4B"}
VNAME = {"f0m0": "Framing 1, J better", "f0m1": "Framing 1, K better", "f1m0": "Framing 2, J better",
         "f1m1": "Framing 2, K better"}
r1 = lambda x: round(float(x), 1)  # noqa: E731
r2 = lambda x: round(float(x), 2)  # noqa: E731
r3 = lambda x: round(float(x), 3)  # noqa: E731


def _w(name, rows):
    json.dump(rows, open(f"{OUT}/{name}.json", "w"), indent=0)
    return len(rows)


def _set_meta(name):
    """(model slug, kind label) for a results/ext set directory name."""
    for slug in NAMES:
        if name.startswith(slug) or name.startswith("orig_" + slug):
            rest = name.replace("orig_", "").replace(slug, "").strip("_")
            break
    else:
        return None, name
    kind = {"": "Original prompts, original game (rerun to N = 64)",
            "pooled": "Original prompts, fresh re-measurement" + (" plus tail samples" if slug == "glm-5.2" else ""),
            "refilled": "Original prompts, lost samples refilled", "stag": "Original prompts, stag hunt",
            "T0.7": "Original prompts, temperature 0.7", "stag_H3": "Original prompts, stag hunt, 3-round memory"}.get(rest)
    if kind is None and "_f" in rest:
        game, f = rest.split("_f")
        kind = f"Flagged wording: {FRAMING_NAMES[int(f)]} ({'stag hunt' if game == 'stag' else 'original game'})"
    return slug, kind or rest


def build():
    os.makedirs(OUT, exist_ok=True)
    counts = {}
    pins = json.load(open("results/providers/pins.json"))
    prov = []
    for r in csv.DictReader(open("results/providers/summary.csv")):
        prov.append({"model": NAMES[r["model"]], "provider": r["provider"], "pinned": pins.get(r["model"]) == r["provider"],
                     "valid_mass": r3(r["valid_mass"]), "mad_vs_adaption": r3(r["mad_vs_adaption"]),
                     "corr_vs_adaption": None if r["corr_vs_adaption"] in ("", "None") else r3(r["corr_vs_adaption"]),
                     "coupling_pp": r1(r["coupling_pp"]), "size_pp": r1(r["size_pp"])})
    counts["ext_providers"] = _w("ext_providers", prov)

    g = load_grid()
    base = [r for r in g if r["H"] == 2 and r["temperature"] == 1.0]
    counts["ext_mechanism"] = _w("ext_mechanism", [
        {"model": NAMES[r["model"]], "deployment": r["deployment"], "pinned": r["pinned"],
         "game": "Stag hunt" if r["game"] == "stag" else "Original game", "wording": FRAMING_NAMES[r["framing"]],
         "variant": f"f{r['framing']}m{r['mapping']}", "size_pp": r1(r["size32_pp"]), "coupling_pp": r1(r["coupling32_pp"]),
         "responsive": r["responsive"]} for r in base])

    resp = []
    for (m, d, game) in sorted({(r["model"], r["deployment"], r["game"]) for r in base}):
        rs = [r for r in base if (r["model"], r["deployment"], r["game"]) == (m, d, game)]
        resp.append({"model": NAMES[m], "deployment": d, "pinned": rs[0]["pinned"],
                     "game": "Stag hunt" if game == "stag" else "Original game",
                     "responsive": sum(r["responsive"] for r in rs), "variants": len(rs),
                     "mean_size_pp": r1(np.mean([r["size32_pp"] for r in rs])),
                     "mean_coupling_pp": r1(np.mean([r["coupling32_pp"] for r in rs]))})
    counts["ext_responsive"] = _w("ext_responsive", resp)

    fac = []
    pinned = [r for r in g if r["pinned"]]

    def add(factor, level, rows, order):
        if rows:
            fac.append({"factor": factor, "level": level, "order": order, "responsive": sum(r["responsive"] for r in rows),
                        "variants": len(rows), "share": r3(np.mean([r["responsive"] for r in rows])),
                        "mean_coupling_pp": r1(np.mean([r["coupling32_pp"] for r in rows])),
                        "mean_p_incumbent": r3(np.mean([r["p_incumbent"] for r in rows]))})
    b2 = [r for r in pinned if r["H"] == 2 and r["temperature"] == 1.0]
    for i, game in enumerate(("coord", "stag")):
        add("Game", "Stag hunt" if game == "stag" else "Original game", [r for r in b2 if r["game"] == game], i)
    for f in range(8):
        add("Wording", FRAMING_NAMES[f], [r for r in b2 if r["framing"] == f], f)
    add("Better option's letter", "J", [r for r in b2 if r["a_letter"] == "J"], 0)
    add("Better option's letter", "K", [r for r in b2 if r["a_letter"] == "K"], 1)
    add("Better option's position", "Listed first", [r for r in b2 if r["a_first"]], 0)
    add("Better option's position", "Listed second", [r for r in b2 if not r["a_first"]], 1)
    for H in (1, 2, 3):
        add("Memory (rounds)", str(H), [r for r in pinned if r["H"] == H and r["framing"] in (0, 1) and r["temperature"] == 1.0], H)
    for i, T in enumerate((0.5, 0.75, 1.0, 1.5)):
        add("Temperature", f"{T:g}", [r for r in pinned if r["H"] == 2 and r["temperature"] == T], i)
    counts["ext_factors"] = _w("ext_factors", fac)

    labels = []
    for m in NAMES:
        for game in ("coord", "stag"):
            rs = [r for r in b2 if r["model"] == m and r["game"] == game]
            if rs:
                f = lambda pred: r3(np.mean([r["p_incumbent"] for r in rs if pred(r)]))  # noqa: E731
                labels.append({"model": NAMES[m], "game": "Stag hunt" if game == "stag" else "Original game",
                               "letter_J": f(lambda r: r["a_letter"] == "J"), "letter_K": f(lambda r: r["a_letter"] == "K"),
                               "listed_first": f(lambda r: r["a_first"]), "listed_second": f(lambda r: not r["a_first"])})
    counts["ext_labels"] = _w("ext_labels", labels)

    sets, ncurve, dose, pervar = [], [], [], []
    for d in sorted(glob.glob("results/ext/*/")):
        name = os.path.basename(d.rstrip("/"))
        slug, kind = _set_meta(name)
        if slug is None:
            continue
        try:
            s = summarize_set(d.rstrip("/"))
        except Exception as e:  # incomplete set
            print("skip", name, e)
            continue
        by = {r["N"]: r for r in s["by_N"]}
        if 32 not in by:
            continue
        row = {"set": name, "model": NAMES[slug], "kind": kind, "variants": len(s["variants"]),
               "coupling32_pp": r1(100 * by[32]["coupling"]["mean"]), "coupling32_lo_pp": r1(100 * by[32]["coupling"]["lo"]),
               "coupling32_hi_pp": r1(100 * by[32]["coupling"]["hi"]),
               "size32_pp": r1(100 * np.mean([v["size32"] for v in s["per_variant"]])),
               "size64_pp": r1(100 * np.mean([v["size64"] for v in s["per_variant"]])),
               "coupling64_pp": r1(100 * by[64]["coupling"]["mean"]) if 64 in by else None}
        if "bootstrap" in s:
            b = s["bootstrap"]["primary"]
            # bounds to 2 decimals: several sit within 0.05 points of zero, where 1 decimal hides the sign
            row.update({"l2_pp": r1(100 * b["mean"]), "l2_lo_pp": r2(100 * b["lo"]), "l2_hi_pp": r2(100 * b["hi"]),
                        "decision": "effect" if (b["lo"] > 0 or b["hi"] < 0) else "no effect"})
        else:
            row.update({"l2_pp": None, "l2_lo_pp": None, "l2_hi_pp": None, "decision": "layer 1 only"})
        sets.append(row)
        for r in s["by_N"]:
            ncurve.append({"set": name, "model": NAMES[slug], "kind": kind, "N": r["N"],
                           "x_independent": r3(r["x_independent"]), "x_coupled": r3(r["x_coupled"])})
        for r in s["dose"]:
            dose.append({"set": name, "model": NAMES[slug], "kind": kind, "N": r["N"],
                         "alpha": r["alpha"], "coupling_pp": r1(100 * r["coupling"]["mean"])})
        for v in s["per_variant"]:
            pervar.append({"set": name, "model": NAMES[slug], "kind": kind, "variant": v["variant"],
                           "size32_pp": r1(100 * v["size32"]), "coupling32_pp": r1(100 * v["coupling32"]),
                           "block32_pp": r1(100 * v["block32"])})
    counts["ext_sets"] = _w("ext_sets", sets)
    counts["ext_ncurve"] = _w("ext_ncurve", ncurve)
    counts["ext_dose"] = _w("ext_dose", dose)
    counts["ext_set_variants"] = _w("ext_set_variants", pervar)

    rt = []
    for p in sorted(glob.glob("results/ext/retest_*.json")):
        r = json.load(open(p))
        slug = os.path.basename(p)[7:-5]
        rt.append({"model": NAMES[slug], "prompts": r["prompts"], "sum_z2": r["sum_z2"], "critical_99": r["critical_99"],
                   "consistent": r["consistent"], "abs_z_gt_2": r["abs_z_gt_2"], "mean_abs_diff": r["mean_abs_diff"]})
    counts["ext_retest"] = _w("ext_retest", rt)
    return counts


if __name__ == "__main__":
    print(build())
