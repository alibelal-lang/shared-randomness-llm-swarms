"""Amendment 3, A and E: run each confirmation analysis as soon as its Adaption policy lands (one at a time).

Run from the project root: .venv/bin/python -u logs/confirm_watcher.py
"""

import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, ".")
from swarm.retest import retest  # noqa: E402
from swarm.topup import merge  # noqa: E402

V2 = "policies/adaption_v2"
ORIG4 = ["f0m0", "f0m1", "f1m0", "f1m1"]


def wait(path):
    while not os.path.exists(path):
        time.sleep(30)
    time.sleep(2)
    meta = json.load(open(path))["meta"]
    lost = sum(r.get("counts", {}).get("errors", 0) for r in json.load(open(path))["records"])
    print(f"[ready] {path}: {meta['calls']} calls, {meta['errors']} failed prompts, {lost} samples lost to API errors, "
          f"cost {meta['total_cost']:.3f}", flush=True)
    return meta


def confirm(policy, out, variants, boot=True):
    if os.path.exists(f"{out}/bootstrap.json") or (not boot and os.path.isdir(out)):
        print(f"[skip] {out} exists", flush=True)
        return
    env = {**os.environ, **({} if boot else {"NOBOOT": "1"})}
    t = time.time()
    r = subprocess.run(["logs/run_confirm3.sh", policy, out, *variants], env=env, capture_output=True, text=True)
    print(f"[confirm] {out} ({time.time() - t:.0f}s)\n{r.stdout[-2500:]}{r.stderr[-800:]}", flush=True)


def retest_and_pool(model, orig, fresh, extra, pooled):
    res = retest(orig, fresh)
    os.makedirs("results/ext", exist_ok=True)
    json.dump(res, open(f"results/ext/retest_{model}.json", "w"), indent=1)
    print(f"[retest] {model}: sum z^2 {res['sum_z2']} vs {res['critical_99']}, consistent={res['consistent']}, "
          f"|z|>2 on {res['abs_z_gt_2']}/{res['prompts']}, mean |diff| {res['mean_abs_diff']}", flush=True)
    parts = ([orig, fresh] if res["consistent"] else [fresh]) + extra
    if len(parts) == 1:
        shutil.copy(parts[0], pooled)
    else:
        merge(parts, pooled)
    return res


def main():
    # A: Qwen 3.8 retest and pooled confirmation
    wait(f"{V2}/qwen3.8-27b_retest.json")
    retest_and_pool("qwen3.8-27b", "policies/adaption_qwen3.8-27b.json", f"{V2}/qwen3.8-27b_retest.json", [],
                    f"{V2}/qwen3.8-27b_pooled.json")
    confirm(f"{V2}/qwen3.8-27b_pooled.json", "results/ext/qwen3.8-27b_pooled", ORIG4)

    # A: refills for Qwen 3.5 397B and Gemma 4 31B
    wait(f"{V2}/glm-5.2_tails.json")
    for m in ("qwen3.5-397b", "gemma-4-31b-it"):
        wait(f"{V2}/{m}_refill.json")
        merge([f"policies/adaption_{m}.json", f"{V2}/{m}_refill.json"], f"{V2}/{m}_refilled.json")
        confirm(f"{V2}/{m}_refilled.json", f"results/ext/{m}_refilled", ORIG4)

    # A: GLM retest, pooled with the tail samples
    wait(f"{V2}/glm-5.2_retest.json")
    retest_and_pool("glm-5.2", "policies/adaption_glm-5.2.json", f"{V2}/glm-5.2_retest.json",
                    [f"{V2}/glm-5.2_tails.json"], f"{V2}/glm-5.2_pooled.json")
    confirm(f"{V2}/glm-5.2_pooled.json", "results/ext/glm-5.2_pooled", ORIG4)

    # D1: stag hunt on the original prompt set
    for m in ("qwen3.8-27b", "glm-5.2"):
        wait(f"{V2}/{m}_stag.json")
        confirm(f"{V2}/{m}_stag.json", f"results/ext/{m}_stag", ORIG4)

    # D2: flagged wordings
    for p in json.load(open("results/explore/d2_plan.json")):
        wait(p["policy"])
        name = os.path.basename(p["policy"])[:-5]
        confirm(p["policy"], f"results/ext/{name}", p["variants"])

    # D3: temperature 0.7
    wait(f"{V2}/qwen3.8-27b_T0.7.json")
    confirm(f"{V2}/qwen3.8-27b_T0.7.json", "results/ext/qwen3.8-27b_T0.7", ORIG4)

    # D4: memory H = 3
    wait(f"{V2}/qwen3.8-27b_stag_H3.json")
    confirm(f"{V2}/qwen3.8-27b_stag_H3.json", "results/ext/qwen3.8-27b_stag_H3", ORIG4)

    # D5: other models, stag hunt
    for m in ("qwen3.5-397b", "gemma-4-31b-it", "gemma-4-26b-a4b-it"):
        wait(f"{V2}/{m}_stag.json")
        confirm(f"{V2}/{m}_stag.json", f"results/ext/{m}_stag", ORIG4)
    print("[watcher] all confirmations done", flush=True)


if __name__ == "__main__":
    main()
