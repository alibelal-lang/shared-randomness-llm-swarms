"""Command line entry point.

  python -m swarm reference
  python -m swarm extract --model qwen/qwen3.6-35b-a3b --out policies/qwen36.json
  python -m swarm show-policy policies/qwen36.json
  python -m swarm simulate --policy policies/qwen36.json --out results/qwen36_pilot
"""

import argparse
import csv
import json
import os
import platform
import sys
import time
from dataclasses import asdict


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def cmd_reference(args):
    from .reference import reference_table

    print(f"{'N':>3} {'indep first entry':>18} {'pair-restore first entry':>25} {'pair-restore terminal':>22}")
    for r in reference_table(T=args.T):
        print(f"{r['N']:>3} {r['independent_first_entry']:>18.6f} {r['pair_restore_first_entry']:>25.4f}"
              f" {r['pair_restore_terminal']:>22.4f}")


def cmd_extract(args):
    from .extract import AdaptionSampleBackend, LocalHFBackend, OpenRouterBackend, extract

    repeats = args.repeats
    if args.backend == "adaption":
        backend = AdaptionSampleBackend(args.model, samples=args.samples, temperature=args.sample_temperature,
                                        reasoning_effort=args.reasoning_effort or None,
                                        inner_workers=args.inner_workers)
        workers = args.workers
        repeats = args.repeats if args.repeats_set else 1  # n samples already replicate each prompt
    elif args.backend == "openrouter":
        backend = OpenRouterBackend(args.model, provider=args.provider, top_logprobs=args.top_logprobs,
                                    disable_reasoning=not args.keep_reasoning)
        workers = args.workers
    else:
        backend = LocalHFBackend(args.model, adapter=args.adapter, revision=args.revision, device=args.device)
        workers = 1
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    checkpoint = args.out + ".partial.jsonl"
    data = extract(backend, H=args.H, framings=args.framings, mappings=args.mappings, repeats=repeats,
                   workers=workers, max_calls=args.max_calls, checkpoint=checkpoint, game=args.game)
    json.dump(data, open(args.out, "w"), indent=1)
    if not data["meta"]["errors"] and os.path.exists(checkpoint):
        os.remove(checkpoint)
    m = data["meta"]
    print(f"Saved {args.out}: {m['calls']} calls, {m['errors']} errors, {m['seconds']}s, "
          f"cost ${m['total_cost']:.5f}, providers {m['providers']}")
    if m["errors"]:
        print("Some calls failed; see records with an 'error' field.", file=sys.stderr)


def cmd_show_policy(args):
    from .extract import load_policy
    import json as _json

    meta = _json.load(open(args.policy))["meta"]
    print(f"{meta['model']} via {meta['backend']} | providers {meta.get('providers')} | T = {args.temperature}")
    combos = [(f, m) for f in meta["framings"] for m in meta["mappings"]]
    tables = {c: load_policy(args.policy, *c, temperature=args.temperature) for c in combos}
    space = next(iter(tables.values()))[1]
    head = "".join(f"  f{f}m{m}" for f, m in combos)
    print(f"\nP(A) by history (own+partner per round, oldest first; A = better option)\n{'history':<8}{head}")
    for code, key in enumerate(space.keys()):
        print(f"{key:<8}" + "".join(f"{tables[c][0][code]:>7.3f}" for c in combos))
    print("\nDiagnostics")
    for c in combos:
        d = tables[c][2]
        extra = (f", {d['samples']} samples, invalid answers {d['invalid_rate']:.1%}" if d["samples"] else
                 f", max repeat spread {d['max_repeat_spread']:.4f}, bounded histories {len(d['bounded_histories'])}")
        print(f"  f{c[0]}m{c[1]}: valid label mass mean {d['mean_valid_mass']:.3f} (min {d['min_valid_mass']:.3f})"
              + extra)


def cmd_compare(args):
    """Agreement between two extracted policies, e.g. Adaption sampled frequencies vs exact logprobs."""
    import math

    import numpy as np

    from .extract import load_policy

    meta_a, meta_b = (json.load(open(p))["meta"] for p in (args.a, args.b))
    combos = [(f, m) for f in meta_a["framings"] for m in meta_a["mappings"]
              if f in meta_b["framings"] and m in meta_b["mappings"]]
    n = meta_a.get("samples_per_prompt") or meta_b.get("samples_per_prompt")
    print(f"A: {meta_a['model']} via {meta_a['backend']}   B: {meta_b['model']} via {meta_b['backend']}")
    for f, m in combos:
        pa, space, _ = load_policy(args.a, f, m)
        pb, _, _ = load_policy(args.b, f, m)
        d = pa - pb
        line = (f"f{f}m{m}: corr {np.corrcoef(pa, pb)[0, 1]:.3f}, mean |diff| {np.abs(d).mean():.3f}, "
                f"max |diff| {np.abs(d).max():.3f} at {space.keys()[int(np.abs(d).argmax())]}, mean diff {d.mean():+.3f}")
        if n:  # sampling error of the sampled side only; provider/extraction noise is extra
            z = d / np.sqrt(np.clip(pb * (1 - pb), 1e-4, None) / n)
            line += f", |z|>2 on {int((np.abs(z) > 2).sum())}/{len(z)} histories"
        print(line)
        if args.verbose:
            for code, key in enumerate(space.keys()):
                print(f"    {key:<7} {pa[code]:.3f}  {pb[code]:.3f}  {d[code]:+.3f}")


def cmd_bootstrap(args):
    from .analysis import bootstrap, contrasts_from_bootstrap

    t0 = time.time()
    variants = [(int(v[1]), int(v[3])) for v in args.variants]
    x, meta = bootstrap(args.policy, B=args.B, seeds_per=args.seeds_per, processes=args.processes, init=args.init,
                        variants=variants)
    res = {"policy": args.policy, "meta": meta, "seconds": round(time.time() - t0, 1),
           "late_x": contrasts_from_bootstrap(x, variants=variants, idx=0),
           "reward": contrasts_from_bootstrap(x, variants=variants, idx=1),
           "replicates": {str(b): {f"f{v[0]}m{v[1]}|N{N}|{s}": val for (v, N, s), val in cells.items()}
                          for b, cells in x.items()}}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"{args.policy}: B={args.B}, {args.seeds_per} seeds per cell per replicate, {res['seconds']}s")
    for metric in ("late_x", "reward"):
        print(f"  [{metric}]")
        for k, v in res[metric].items():
            flag = "  *" if (v["lo"] > 0 or v["hi"] < 0) else ""
            print(f"    {k:<20} {v['mean']:+.3f}  [{v['lo']:+.3f}, {v['hi']:+.3f}]{flag}")


def cmd_live(args):
    import numpy as np

    from .extract import load_policy
    from .game import HistorySpace
    from .live import live_campaign
    from .samplers import parse_sampler
    from .simulate import run_population

    variants = [(int(v[1]), int(v[3])) for v in args.variants]
    out, secs = live_campaign(args.model, variants, args.N, args.runs, args.T, init=args.init)
    os.makedirs(args.out, exist_ok=True)
    decisions = [d | {"framing": f, "mapping": m, "N": N} for (f, m, N, s), xs, dec in out for d in dec]
    _write_csv(f"{args.out}/decisions.csv", decisions)
    space = HistorySpace(2)
    ind = parse_sampler("independent")
    summary = []
    for (f, m) in variants:
        pA, _, _ = load_policy(args.policy, f, m)
        pmap = dict(zip(space.keys(), pA))
        for N in args.N:
            live_x = np.array([xs for (ff, mm, NN, s), xs, _ in out if (ff, mm, NN) == (f, m, N)])
            seeds = [s for (ff, mm, NN, s), _, _ in out if (ff, mm, NN) == (f, m, N)]
            sim_x = np.array([run_population(pA, space, N, args.T, ind, args.init, s, keep_trace=True)[1]["x"]
                              for s in seeds])
            # many simulated replicates per live seed for a smooth comparator
            sim_many = np.array([run_population(pA, space, N, args.T, ind, args.init, 50000 + k, keep_trace=True)[1]["x"]
                                 for k in range(500)])
            dec = [d for d in decisions if (d["framing"], d["mapping"], d["N"]) == (f, m, N)]
            act = np.array([d["action"] for d in dec], float)
            pex = np.array([pmap[d["history"]] for d in dec])
            late = slice(args.T * 2 // 3, None)
            row = {"framing": f, "mapping": m, "N": N, "runs": len(seeds), "decisions": len(dec),
                   "live_mean_action": act.mean(), "extracted_mean_p": pex.mean(),
                   "calibration_gap": act.mean() - pex.mean(),
                   "calibration_gap_se": float(np.sqrt(np.mean(pex * (1 - pex)) / len(dec))),
                   "live_late_x": float(live_x[:, late].mean()), "sim_late_x_same_seeds": float(sim_x[:, late].mean()),
                   "sim_late_x_500": float(sim_many[:, late].mean()),
                   "sim_late_x_500_sd_of_20run_mean": float(sim_many[:, late].mean(1).std() / np.sqrt(len(seeds))),
                   "fallback_rate": float(np.mean([d["fallback"] for d in dec])),
                   "retry_rate": float(np.mean([d["attempts"] > 1 for d in dec])),
                   "cost": float(sum(d["cost"] for d in dec))}
            summary.append(row)
            _write_csv(f"{args.out}/trace_f{f}m{m}_N{N}.csv",
                       [{"round": t + 1, "live_x": float(live_x[:, t].mean()), "sim_x": float(sim_many[:, t].mean())}
                        for t in range(args.T)])
    _write_csv(f"{args.out}/summary.csv", summary)
    print(f"Live campaign {args.model}: {len(decisions)} decisions in {secs:.0f}s, "
          f"cost {sum(r['cost'] for r in summary):.4f} credits")
    for r in summary:
        print(f"  f{r['framing']}m{r['mapping']} N={r['N']:>2}: live mean action {r['live_mean_action']:.3f} vs extracted p "
              f"{r['extracted_mean_p']:.3f} (gap {r['calibration_gap']:+.3f} ± {1.96 * r['calibration_gap_se']:.3f}); "
              f"late x live {r['live_late_x']:.3f} vs sim {r['sim_late_x_500']:.3f} "
              f"(± {1.96 * r['sim_late_x_500_sd_of_20run_mean']:.3f}); fallback {r['fallback_rate']:.1%}")


def cmd_simulate(args):
    import numpy as np

    from .extract import load_policy
    from .samplers import parse_sampler
    from .simulate import paired_contrasts, run_population, summarize

    pA, space, diag = load_policy(args.policy, args.framing, args.mapping, args.temperature)
    samplers = [parse_sampler(s, N0=args.N0) for s in args.samplers]
    t0 = time.time()
    results = [
        run_population(pA, space, N, args.T, smp, args.init, seed, args.threshold,
                       game=diag["meta"].get("game", "coord"))
        for N in args.N
        for smp in samplers
        for seed in range(args.seed_offset, args.seed_offset + args.seeds)
    ]
    summary, contrasts = summarize(results), paired_contrasts(results)

    os.makedirs(args.out, exist_ok=True)
    _write_csv(f"{args.out}/runs.csv", [asdict(r) for r in results])
    _write_csv(f"{args.out}/summary.csv", summary)
    _write_csv(f"{args.out}/contrasts.csv", contrasts)
    manifest = {
        "command": " ".join(sys.argv),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seconds": round(time.time() - t0, 1),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "args": vars(args),
        "policy_diagnostics": diag,
        "pA": dict(zip(space.keys(), map(float, pA))),
    }
    json.dump(manifest, open(f"{args.out}/manifest.json", "w"), indent=1, default=str)

    print(f"{diag['meta']['model']} | framing {args.framing} mapping {args.mapping} T={args.temperature} | "
          f"init {args.init} | {args.seeds} runs per cell, {args.T} rounds, entry x >= {args.threshold}")
    print(f"\n{'N':>3} {'sampler':<22} {'P(entry)':>9} {'95% CI':>15} {'RMST':>6} {'occup':>6} {'final':>6}"
          f" {'reward':>7} {'x late':>7} {'all-A':>6} {'alpha':>6} {'infeas':>6}")
    for s in summary:
        print(f"{s['N']:>3} {s['sampler']:<22} {s['entry_prob']:>9.3f} "
              f"[{s['entry_ci_lo']:.3f},{s['entry_ci_hi']:.3f}] {s['rmst']:>6.1f} {s['occupancy']:>6.3f} "
              f"{s['final_in_prob']:>6.3f} {s['mean_reward']:>7.3f} {s['mean_x_late']:>7.3f} {s['all_A_prob']:>6.3f}"
              f" {s['mean_alpha']:>6.3f} {s['infeasible_frac']:>6.3f}")
    print(f"\nPaired entry difference vs independent (same seeds):")
    for c in contrasts:
        print(f"{c['N']:>3} {c['sampler']:<22} {c['diff']:+.3f} [{c['ci_lo']:+.3f},{c['ci_hi']:+.3f}]")
    print(f"\nSaved to {args.out}/ ({manifest['seconds']}s)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m swarm")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("reference", help="exact synthetic reference table from the research plan")
    r.add_argument("--T", type=int, default=100)
    r.set_defaults(func=cmd_reference)

    e = sub.add_parser("extract", help="extract P(label | history) for every history, framing and mapping")
    e.add_argument("--backend", choices=["adaption", "openrouter", "local"], default="adaption")
    e.add_argument("--samples", type=int, default=200, help="adaption: samples per prompt")
    e.add_argument("--sample-temperature", type=float, default=1.0, help="adaption: sampling temperature")
    e.add_argument("--reasoning-effort", default="none", help="adaption: reasoning effort ('' to omit)")
    e.add_argument("--inner-workers", type=int, default=8, help="adaption: parallel samples per prompt")
    e.add_argument("--model", required=True)
    e.add_argument("--provider", help="pin one OpenRouter provider slug, no fallbacks")
    e.add_argument("--adapter", help="local backend: path to a LoRA adapter")
    e.add_argument("--revision", help="local backend: Hugging Face commit to pin, e.g. 851bf6e806")
    e.add_argument("--device", help="local backend: cpu or mps (cpu is ~3x faster for Qwen3.5 linear attention)")
    e.add_argument("--top-logprobs", type=int, default=20)
    e.add_argument("--keep-reasoning", action="store_true", help="do not ask OpenRouter to disable reasoning")
    e.add_argument("--H", type=int, default=2)
    e.add_argument("--game", choices=["coord", "stag"], default="coord")
    e.add_argument("--framings", type=int, nargs="+", default=[0, 1])
    e.add_argument("--mappings", type=int, nargs="+", default=[0, 1])
    e.add_argument("--repeats", type=int, default=None, help="default 2 (openrouter/local) or 1 (adaption)")
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--max-calls", type=int, default=20000)
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_extract)

    s = sub.add_parser("show-policy", help="print an extracted policy table and diagnostics")
    s.add_argument("policy")
    s.add_argument("--temperature", type=float, default=1.0)
    s.set_defaults(func=cmd_show_policy)

    c = sub.add_parser("compare-policies", help="agreement between two extracted policies")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("-v", "--verbose", action="store_true")
    c.set_defaults(func=cmd_compare)

    b = sub.add_parser("bootstrap", help="policy-estimation bootstrap for the primary contrasts")
    b.add_argument("--policy", required=True)
    b.add_argument("--B", type=int, default=100)
    b.add_argument("--seeds-per", type=int, default=100)
    b.add_argument("--processes", type=int, default=6)
    b.add_argument("--init", default="incumbent_B")
    b.add_argument("--variants", nargs="+", default=["f0m0", "f0m1", "f1m0", "f1m1"])
    b.add_argument("--out", required=True)
    b.set_defaults(func=cmd_bootstrap)

    lv = sub.add_parser("live", help="live population validation through Adaption (independent arm)")
    lv.add_argument("--model", required=True)
    lv.add_argument("--policy", required=True, help="extracted policy for the simulated comparator")
    lv.add_argument("--variants", nargs="+", default=["f0m0", "f0m1", "f1m0", "f1m1"])
    lv.add_argument("--N", type=int, nargs="+", default=[4, 16])
    lv.add_argument("--runs", type=int, default=20)
    lv.add_argument("--T", type=int, default=30)
    lv.add_argument("--init", default="incumbent_B")
    lv.add_argument("--out", required=True)
    lv.set_defaults(func=cmd_live)

    m = sub.add_parser("simulate", help="frozen-policy population sweep")
    m.add_argument("--policy", required=True)
    m.add_argument("--framing", type=int, default=0)
    m.add_argument("--mapping", type=int, default=0)
    m.add_argument("--temperature", type=float, default=1.0)
    m.add_argument("--N", type=int, nargs="+", default=[4, 8, 16, 32])
    m.add_argument("--T", type=int, default=100)
    m.add_argument("--seeds", type=int, default=500)
    m.add_argument("--seed-offset", type=int, default=0)
    m.add_argument("--samplers", nargs="+",
                   default=["independent", "pair_shared:restore", "pair_shared:0.3", "block:2"])
    m.add_argument("--N0", type=int, default=4)
    m.add_argument("--init", default="incumbent_B")
    m.add_argument("--threshold", type=float, default=0.75)
    m.add_argument("--out", required=True)
    m.set_defaults(func=cmd_simulate)

    args = p.parse_args(argv)
    if args.cmd == "extract":
        args.repeats_set = args.repeats is not None
        args.repeats = args.repeats if args.repeats_set else 2
    args.func(args)


if __name__ == "__main__":
    main()
