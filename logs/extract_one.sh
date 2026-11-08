#!/bin/zsh
# Resumable Adaption extraction for one model; skips if the final policy file already exists.
cd "/Users/alibelal/Swarm alignment"
m=$1
[[ -f policies/adaption_$m.json ]] && { echo "=== $m already done"; exit 0; }
echo "=== $m start $(date +%H:%M:%S)"
.venv/bin/python -u -m swarm extract --backend adaption --model $m --samples 200 --workers 4 --inner-workers 16 --out policies/adaption_$m.json 2>&1 | grep --line-buffered -E "progress|resumed|Saved|Error|error|Traceback" | sed -u "s/^/[$m] /"
echo "=== $m end $(date +%H:%M:%S)"
