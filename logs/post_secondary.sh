#!/bin/zsh
# For each secondary model: wait for an error-free Adaption policy, then run confirmatory sweeps and the bootstrap.
cd "/Users/alibelal/Swarm alignment"
for m in "$@"; do
  until [[ -f policies/adaption_$m.json ]] && .venv/bin/python -c "import json,sys; sys.exit(json.load(open('policies/adaption_$m.json'))['meta']['errors'])" 2>/dev/null; do sleep 30; done
  [[ -f results/confirm/$m/bootstrap.json ]] && { echo "=== $m already analysed"; continue; }
  echo "=== $m sweeps start $(date +%H:%M:%S)"
  logs/run_confirm.sh $m | tail -1
  .venv/bin/python -m swarm bootstrap --policy policies/adaption_$m.json --B 100 --seeds-per 100 --processes 4 --out results/confirm/$m/bootstrap.json 2>&1 | sed -n '1,4p'
  echo "=== $m analysis done $(date +%H:%M:%S)"
done
echo "=== all secondary analyses done $(date +%H:%M:%S)"
