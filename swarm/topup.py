"""Extra Adaption samples for chosen prompts, and pooling of sampled policy files that share one prompt set.

  python -m swarm.topup sample --base policies/adaption_glm-5.2.json --target 0 1 AA.AA 5000 --out supp.json
  python -m swarm.topup errors --base policies/adaption_qwen3.5-397b.json --out supp.json   # refill lost samples
  python -m swarm.topup merge policies/a.json supp.json --out pooled.json
"""

import argparse
import os
import json
import time

from .extract import AdaptionSampleBackend
from .game import A, B, HistorySpace
from .prompts import LABEL_MAPPINGS, build_prompt


def _record(s, f, m, key, lab):
    return {"history": key, "framing": f, "mapping": m, "repeat": 0, "label_A": lab[A],
            "mass_A": s["mass"][lab[A]], "mass_B": s["mass"][lab[B]], "found_A": True, "found_B": True,
            "listed_mass": 1.0, "min_listed": 0.0, "top": [], "provider": "adaption",
            "served_model": s.get("served_model"), "prompt_tokens": s.get("prompt_tokens"),
            "completion_tokens": s.get("completion_tokens"), "cost": s.get("cost"), "counts": s["counts"],
            "invalid_examples": s["invalid_examples"]}


def sample(base: str, targets: list, out: str):
    """targets: [(framing, mapping, history key, n samples)]. Prompts and settings are taken from the base file."""
    meta = {**json.load(open(base))["meta"], "top_p": os.environ.get("ADAPTION_TOP_P") or "server default (0.95)"}
    space = HistorySpace(meta["H"])
    game = meta.get("game", "coord")
    recs, t0 = [], time.time()
    for f, m, key, n in targets:
        be = AdaptionSampleBackend(meta["model"], samples=n, temperature=meta["sample_temperature"],
                                   reasoning_effort=meta["reasoning_effort"], inner_workers=16)
        hist = space.histories[space.keys().index(key)]
        s = be.score(build_prompt(hist, f, m, game=game))
        recs.append(_record(s, f, m, key, LABEL_MAPPINGS[m]))
        print(f"f{f}m{m} {key}: {s['counts']} ({time.time() - t0:.0f}s, {be.limiter.status()})", flush=True)
    data = {"meta": {**meta, "samples_per_prompt": "top-up", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "seconds": round(time.time() - t0, 1), "calls": sum(t[3] for t in targets),
                     "errors": 0, "total_cost": sum(r["cost"] or 0 for r in recs), "topup_of": base},
            "records": recs}
    json.dump(data, open(out, "w"), indent=1)
    print(f"Saved {out}")


def merge(paths: list, out: str):
    """Sum answer counts per (framing, mapping, history) across files with the same prompt set; Jeffreys again."""
    datas = [json.load(open(p)) for p in paths]
    hashes = {d["meta"]["prompt_hash"] for d in datas}
    if len(hashes) != 1 or len({d["meta"]["model"] for d in datas}) != 1:
        raise SystemExit(f"cannot pool different prompt sets or models: {hashes}")
    acc = {}
    for d in datas:
        for r in d["records"]:
            if "error" in r or "counts" not in r:
                continue
            k = (r["framing"], r["mapping"], r["history"])
            if k not in acc:
                acc[k] = {**r, "counts": dict(r["counts"]), "cost": r.get("cost") or 0}
            else:
                for c in ("J", "K", "invalid", "errors", "n"):
                    acc[k]["counts"][c] += r["counts"].get(c, 0)
                acc[k]["cost"] += r.get("cost") or 0
    for r in acc.values():
        c, la = r["counts"], r["label_A"]
        lb = "K" if la == "J" else "J"
        r["mass_A"], r["mass_B"] = (c[la] + 0.5) / (c["n"] + 1), (c[lb] + 0.5) / (c["n"] + 1)
    meta = {**datas[0]["meta"], "samples_per_prompt": "pooled", "merged_from": paths,
            "calls": sum(d["meta"]["calls"] for d in datas), "errors": 0,
            "total_cost": sum(d["meta"]["total_cost"] for d in datas),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    json.dump({"meta": meta, "records": list(acc.values())}, open(out, "w"), indent=1)
    ns = sorted(r["counts"]["n"] for r in acc.values())
    print(f"Saved {out}: {len(acc)} prompts, valid samples per prompt min {ns[0]} median {ns[len(ns) // 2]} max {ns[-1]}")


def main():
    p = argparse.ArgumentParser(prog="python -m swarm.topup")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--base", required=True)
    s.add_argument("--target", nargs=4, action="append", required=True, metavar=("F", "M", "HIST", "N"))
    s.add_argument("--out", required=True)
    e = sub.add_parser("errors", help="refill samples lost to API errors so each prompt gets back to full n")
    e.add_argument("--base", required=True)
    e.add_argument("--min-lost", type=int, default=1)
    e.add_argument("--out", required=True)
    g = sub.add_parser("merge")
    g.add_argument("paths", nargs="+")
    g.add_argument("--out", required=True)
    a = p.parse_args()
    if a.cmd == "sample":
        sample(a.base, [(int(f), int(m), h, int(n)) for f, m, h, n in a.target], a.out)
    elif a.cmd == "errors":
        d = json.load(open(a.base))
        t = [(r["framing"], r["mapping"], r["history"], r["counts"]["errors"]) for r in d["records"]
             if r.get("counts", {}).get("errors", 0) >= a.min_lost]
        print(f"{len(t)} prompts to refill, {sum(x[3] for x in t)} samples")
        sample(a.base, t, a.out)
    else:
        merge(a.paths, a.out)


if __name__ == "__main__":
    main()
