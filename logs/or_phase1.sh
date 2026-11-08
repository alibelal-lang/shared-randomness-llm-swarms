#!/bin/zsh
# OpenRouter replication of the scoping prompt set (coord, H=2, framings 0-1, mappings 0-1), one file per provider.
cd "/Users/alibelal/Swarm alignment"
run() {
  local model=$1 prov=$2 short=$3
  local out="policies/openrouter/providers/${short}__${prov//\//-}.json"
  [[ -f $out ]] && { echo "skip $out"; return; }
  .venv/bin/python -u -m swarm extract --backend openrouter --model $model --provider $prov --repeats 1 --workers 4 \
    --out $out > logs/or/phase1_${short}__${prov//\//-}.log 2>&1
  echo "rc=$? $out"
}
while read -r model prov short; do run $model $prov $short & done < logs/or_phase1_jobs.txt
wait
echo "phase1 done"
