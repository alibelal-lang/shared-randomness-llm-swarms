"""Post-hoc robustness checks requested after the draft (8 Nov 2026). Not part of the locked protocol.

  python -m swarm.checks seedsplit   # rerun the 896 explored variants on a fresh seed batch (no model calls)
  python -m swarm.checks starts      # headline GLM policy from A-A and B-B starts: escape vs leaving
  python -m swarm.checks report      # print both summaries

Check 1 (seed split). In the original grid, the size effect (ind32 - ind4) and the coupling effect (res32 - ind32)
share the same ind32 runs with opposite signs, so noise in ind32 alone pushes their correlation negative, and
selecting "stuck" variants on a low ind32 inflates the coupling effect measured from that same ind32. Here batch A
is the original grid (seeds 0-199) and batch B is a fresh run of ind4, ind32 and res32 on seeds 200000-200199,
which no other analysis uses. Size is taken from one batch and coupling from the other.

Check 2 (starts). The confirmed GLM 5.2 stag-hunt policy (AI agents wording, f4m2) simulated exactly as in the
confirmation (seeds 0-499, T = 100), from all agents remembering two rounds of B-B or two rounds of A-A.
Population state uses hysteresis on the share x_t playing A, over K consecutive rounds:
  at A: x_t >= 0.75 for K rounds (0.75 = the share above which A is the best reply in the stag hunt)
  left A: x_t < 0.75 for K rounds after being at A
  at B: x_t <= 0.25 for K rounds
"""

import csv
import json
import os
import sys
from multiprocessing import get_context

import numpy as np

from .extract import load_policy
from .game import HistorySpace
from .samplers import parse_sampler
from .simulate import run_population

OUT = "results/checks"
SEEDS_B = range(200000, 200200)
CELLS = [(4, "independent"), (32, "independent"), (32, "pair_shared:restore")]
GLM = "policies/adaption_v2/glm-5.2_stag_f4.json"
K = 10


# ---------------------------------------------------------------- check 1: seed split

def _cell(args):
    key, pA, N, spec, game = args
    space, smp = HistorySpace(2), parse_sampler(spec)
    xs = [run_population(pA, space, N, 100, smp, "incumbent_B", s, game=game).mean_x_late for s in SEEDS_B]
    return key, float(np.mean(xs))


def seedsplit():
    import glob
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for path in sorted(glob.glob("policies/openrouter/grid/*__H2.json")):
        model, dep, game, _ = os.path.basename(path)[:-5].split("__")
        meta = json.load(open(path))["meta"]
        for f in meta["framings"]:
            for m in meta["mappings"]:
                pA, _, _ = load_policy(path, f, m)
                for N, s in CELLS:
                    jobs.append(((model, dep, game, f, m, N, s), pA, N, s, game))
    print(f"{len(jobs)} cells", flush=True)
    res = {}
    with get_context("fork").Pool(max(1, (os.cpu_count() or 4) - 1)) as pool:
        for i, (key, x) in enumerate(pool.imap_unordered(_cell, jobs, chunksize=4)):
            res[key] = x
            if (i + 1) % 300 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    rows = {}
    for (model, dep, game, f, m, N, s), x in res.items():
        r = rows.setdefault((model, dep, game, f, m), {"model": model, "deployment": dep, "game": game,
                                                       "framing": f, "mapping": m})
        r[f"x_{'ind' if s == 'independent' else 'res'}{N}_B"] = round(x, 5)
    with open(f"{OUT}/seedsplit_batchB.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(next(iter(rows.values())).keys()))
        w.writeheader()
        w.writerows(sorted(rows.values(), key=lambda r: tuple(r.values())))
    print(f"wrote {OUT}/seedsplit_batchB.csv ({len(rows)} variants)")


def seedsplit_table():
    """One row per variant with size and coupling from each batch (points)."""
    from .explore_report import load as load_grid
    a = {(r["model"], r["deployment"], r["game"], int(r["framing"]), int(r["mapping"])): r
         for r in load_grid() if r["H"] == 2 and r["temperature"] == 1.0}
    out = []
    for r in csv.DictReader(open(f"{OUT}/seedsplit_batchB.csv")):
        k = (r["model"], r["deployment"], r["game"], int(r["framing"]), int(r["mapping"]))
        ra = a[k]
        i4b, i32b, r32b = (float(r[c]) for c in ("x_ind4_B", "x_ind32_B", "x_res32_B"))
        out.append({"key": k, "game": r["game"],
                    # batch A from the grid's unrounded effects (as in the paper), not its rounded x columns
                    "size_A": ra["size32_pp"], "coup_A": ra["coupling32_pp"],
                    "size_B": 100 * (i32b - i4b), "coup_B": 100 * (r32b - i32b)})
    return out


def _summ(size, coup):
    size, coup = np.asarray(size), np.asarray(coup)
    stuck, up, flat = size <= -2, size >= 2, np.abs(size) < 2
    rec = coup[stuck] / -size[stuck]
    return {"r": round(float(np.corrcoef(size, coup)[0, 1]), 3), "stuck": int(stuck.sum()),
            "stuck_coup_pos": int((coup[stuck] > 0).sum()), "stuck_coup_mean": round(float(coup[stuck].mean()), 2),
            "stuck_recovery_median": round(float(np.median(rec)), 3),
            "stuck_recovery_iqr": [round(float(q), 3) for q in np.percentile(rec, [25, 75])],
            "better": int(up.sum()), "better_coup_neg": int((coup[up] < 0).sum()),
            "better_coup_mean": round(float(coup[up].mean()), 2) if up.any() else None,
            "flat": int(flat.sum()), "flat_abs_coup_lt1": round(float(np.mean(np.abs(coup[flat]) < 1)), 3)}


def seedsplit_summary():
    t = seedsplit_table()
    g = lambda k: [r[k] for r in t]  # noqa: E731
    s = {"same_A (original)": _summ(g("size_A"), g("coup_A")), "same_B": _summ(g("size_B"), g("coup_B")),
         "size_A_coup_B": _summ(g("size_A"), g("coup_B")), "size_B_coup_A": _summ(g("size_B"), g("coup_A"))}
    # size effects replicate across batches?
    s["size_A_vs_size_B_r"] = round(float(np.corrcoef(g("size_A"), g("size_B"))[0, 1]), 3)
    s["stuck_both"] = int(sum(r["size_A"] <= -2 and r["size_B"] <= -2 for r in t))
    json.dump(s, open(f"{OUT}/seedsplit_summary.json", "w"), indent=1)
    return s


# ---------------------------------------------------------------- check 2: A-A vs B-B starts

def _runs(x, thr_hi=0.75, thr_lo=0.25, k=K):
    """Hysteresis events on one trajectory."""
    hi, lo = x >= thr_hi, x <= thr_lo

    def first_run(mask, start=0):
        c = 0
        for t in range(start, len(mask)):
            c = c + 1 if mask[t] else 0
            if c >= k:
                return t - k + 1
        return None
    return first_run, hi, lo


def _events(x, init):
    first_run, hi, lo = _runs(x)
    out = {}
    if init == "incumbent_B":
        t_a = first_run(hi)
        out["reach_A"] = t_a is not None
        out["t_reach_A"] = t_a
        out["left_A_after_reaching"] = (t_a is not None) and first_run(~hi, t_a + K) is not None
    else:
        t_leave = first_run(~hi)
        out["left_A"] = t_leave is not None
        out["t_left_A"] = t_leave
        out["reach_B"] = first_run(lo) is not None
    return out


def _start_cell(args):
    pA, game, N, spec, init = args
    space, smp = HistorySpace(2), parse_sampler(spec)
    rows = []
    for s in range(500):
        res, tr = run_population(pA, space, N, 100, smp, init, s, keep_trace=True, game=game)
        rows.append({"N": N, "sampler": spec, "init": init, "seed": s, "x_late": res.mean_x_late,
                     "x_mean": float(tr["x"].mean()), **_events(tr["x"], init)})
    return rows


def starts():
    os.makedirs(OUT, exist_ok=True)
    pA, _, _ = load_policy(GLM, 4, 2)
    game = json.load(open(GLM))["meta"]["game"]
    jobs = [(pA, game, N, spec, init) for N in (4, 8, 16, 32, 64)
            for spec in ("independent", "pair_shared:restore") for init in ("incumbent_B", "incumbent_A")]
    rows = []
    with get_context("fork").Pool(max(1, (os.cpu_count() or 4) - 1)) as pool:
        for rs in pool.imap_unordered(_start_cell, jobs):
            rows += rs
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(f"{OUT}/starts_glm_stag_f4.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}/starts_glm_stag_f4.csv ({len(rows)} runs)")


def starts_summary():
    rows = list(csv.DictReader(open(f"{OUT}/starts_glm_stag_f4.csv")))
    b = lambda v: v == "True"  # noqa: E731
    out = []
    for N in (4, 8, 16, 32, 64):
        for spec in ("independent", "pair_shared:restore"):
            rB = [r for r in rows if int(r["N"]) == N and r["sampler"] == spec and r["init"] == "incumbent_B"]
            rA = [r for r in rows if int(r["N"]) == N and r["sampler"] == spec and r["init"] == "incumbent_A"]
            reached = [r for r in rB if b(r["reach_A"])]
            out.append({
                "N": N, "sampler": spec,
                "from_B_reach_A": round(len(reached) / len(rB), 3),
                "from_B_median_round": float(np.median([int(r["t_reach_A"]) for r in reached])) if reached else None,
                "from_B_then_left_A": round(np.mean([b(r["left_A_after_reaching"]) for r in reached]), 3) if reached else None,
                "from_A_left_A": round(np.mean([b(r["left_A"]) for r in rA]), 3),
                "from_A_reach_B": round(np.mean([b(r["reach_B"]) for r in rA]), 3),
                "late_x_from_B": round(np.mean([float(r["x_late"]) for r in rB]), 3),
                "late_x_from_A": round(np.mean([float(r["x_late"]) for r in rA]), 3),
            })
    json.dump(out, open(f"{OUT}/starts_summary.json", "w"), indent=1)
    return out


def _start_variant(args):
    key, path, f, m, game = args
    pA, space, _ = load_policy(path, f, m)
    out = {}
    for N, spec in ((4, "independent"), (32, "independent"), (32, "pair_shared:restore")):
        smp = parse_sampler(spec)
        for init in ("incumbent_B", "incumbent_A"):
            late, ev = [], []
            for s in range(200):
                res, tr = run_population(pA, space, N, 100, smp, init, s, keep_trace=True, game=game)
                late.append(res.mean_x_late)
                ev.append(_events(tr["x"], init))
            tag = f"{'ind' if spec == 'independent' else 'res'}{N}_{init[-1]}"
            out[f"x_{tag}"] = float(np.mean(late))
            out[f"{'reachA' if init == 'incumbent_B' else 'leftA'}_{tag}"] = float(np.mean(
                [e["reach_A"] if init == "incumbent_B" else e["left_A"] for e in ev]))
    return key, out


def starts_all():
    """Every stuck explored variant (batch A), from A-A and B-B starts, seeds 0-199."""
    from .explore_report import load as load_grid
    stuck = [r for r in load_grid() if r["H"] == 2 and r["temperature"] == 1.0 and r["responsive"]]
    jobs = [((r["model"], r["deployment"], r["game"], int(r["framing"]), int(r["mapping"])),
             f"policies/openrouter/grid/{r['model']}__{r['deployment']}__{r['game']}__H2.json",
             int(r["framing"]), int(r["mapping"]), r["game"]) for r in stuck]
    rows = []
    with get_context("fork").Pool(max(1, (os.cpu_count() or 4) - 1)) as pool:
        for key, out in pool.imap_unordered(_start_variant, jobs):
            rows.append({"model": key[0], "deployment": key[1], "game": key[2], "framing": key[3], "mapping": key[4],
                         **{k: round(v, 4) for k, v in out.items()}})
    with open(f"{OUT}/starts_stuck_variants.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}/starts_stuck_variants.csv ({len(rows)} variants)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "seedsplit":
        seedsplit()
        print(json.dumps(seedsplit_summary(), indent=1))
    elif cmd == "starts_all":
        starts_all()
    elif cmd == "starts":
        starts()
        for r in starts_summary():
            print(r)
    else:
        print(json.dumps(seedsplit_summary(), indent=1))
        for r in starts_summary():
            print(r)
