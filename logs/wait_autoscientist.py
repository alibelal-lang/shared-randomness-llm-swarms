"""Wait for both AutoScientist runs, then download and unpack each best checkpoint."""

import json
import os
import tarfile
import time

from dotenv import load_dotenv

load_dotenv(".env", override=True)
from adaption import Adaption  # noqa: E402

c = Adaption()
info = json.load(open("data/learning/autoscientist_runs.json"))
pending = dict(info["runs"])
t0 = time.time()
last = {}
while pending:
    for arm, rid in list(pending.items()):
        r = c.autoscientist.get(rid).model_dump()
        state = (r["status"], r.get("iterations_completed"), r.get("best_win_rate"))
        if state != last.get(arm):
            print(f"[{time.time() - t0:6.0f}s] {arm}: status={r['status']} iterations={r.get('iterations_completed')} "
                  f"best_win_rate={r.get('best_win_rate')} download={r.get('download_available')} error={r.get('error')}",
                  flush=True)
            last[arm] = state
        if r["status"] in ("succeeded", "failed", "cancelled"):
            json.dump(r, open(f"data/learning/run_{arm}.json", "w"), indent=1, default=str)
            if r["status"] == "succeeded" and r.get("download_available"):
                os.makedirs(f"models/adapters/{arm}", exist_ok=True)
                tgz = f"models/adapters/{arm}.tgz"
                with c.autoscientist.with_streaming_response.download(rid) as resp:
                    resp.stream_to_file(tgz)
                with tarfile.open(tgz) as tf:
                    tf.extractall(f"models/adapters/{arm}", filter="data")
                print(f"{arm}: downloaded {os.path.getsize(tgz) / 1e6:.1f} MB -> models/adapters/{arm}", flush=True)
            pending.pop(arm)
    if pending:
        time.sleep(60)
print("ALL AUTOSCIENTIST RUNS FINISHED", flush=True)
