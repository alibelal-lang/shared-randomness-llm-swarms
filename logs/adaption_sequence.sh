#!/bin/zsh
# Rate-limited Adaption sequence: primary extraction -> live validation -> secondary extractions.
cd "/Users/alibelal/Swarm alignment"
export ADAPTION_MAX_RATE=${ADAPTION_MAX_RATE:-5}
ex() {  # model samples
  [[ -f policies/adaption_$1.json ]] && { echo "=== $1 already extracted"; return; }
  echo "=== extract $1 ($2 samples) start $(date +%H:%M:%S)"
  .venv/bin/python -u -m swarm extract --backend adaption --model $1 --samples $2 --workers 4 --inner-workers 8 --out policies/adaption_$1.json 2>&1 | grep --line-buffered -E "progress|resumed|Saved|Error|error|Traceback" | sed -u "s/^/[$1] /"
}
ex qwen3.8-27b 200
echo "=== live validation start $(date +%H:%M:%S)"
.venv/bin/python -u -m swarm live --model qwen3.8-27b --policy policies/adaption_qwen3.8-27b.json --variants f0m0 f1m1 --N 4 16 --runs 20 --T 30 --out results/live/qwen3.8-27b 2>&1 | grep --line-buffered -v "^$" | sed -u "s/^/[live] /"
for m in gemma-4-31b-it glm-5.2 qwen3.5-397b gemma-4-26b-a4b-it; do ex $m 100; done
echo "=== sequence done $(date +%H:%M:%S)"
