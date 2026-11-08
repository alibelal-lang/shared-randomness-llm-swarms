#!/bin/zsh
# Confirmatory frozen-policy sweeps per PROTOCOL.md: seeds 0-499, all variants, three starts.
# Usage: logs/run_confirm.sh <model-slug> [more slugs]   (policy file: policies/adaption_<slug>.json)
cd "/Users/alibelal/Swarm alignment"
for slug in "$@"; do
  for init in incumbent_B empty split; do
    for f in 0 1; do for m in 0 1; do
      echo "$slug $init $f $m"
    done; done
  done
done | xargs -P 4 -L 1 zsh -c '.venv/bin/python -m swarm simulate --policy policies/adaption_$0.json --framing $2 --mapping $3 --init $1 --seeds 500 --seed-offset 0 --out results/confirm/$0/f$2m$3_$1 > /dev/null && echo "done $0 f$2m$3 $1"'
