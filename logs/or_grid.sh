#!/bin/zsh
# OpenRouter exploration grid (Amendment 3, C). One extract per deployment x game x H.
cd "/Users/alibelal/Swarm alignment"
one() {
  IFS='|' read -r model short prov game H framings <<< "$1"
  local out="policies/openrouter/grid/${short}__${prov//\//-}__${game}__H${H}.json"
  [[ -f $out ]] && { echo "skip $out"; return 0; }
  .venv/bin/python -u -m swarm extract --backend openrouter --model $model --provider $prov --repeats 1 --workers 4 \
    --game $game --H $H --framings ${=framings} --mappings 0 1 2 3 --out $out > logs/or/grid_$(basename $out .json).log 2>&1
  echo "rc=$? $(date -u +%H:%M:%S) $out $(grep -o 'errors [0-9]*' logs/or/grid_$(basename $out .json).log | tail -1)"
}
typeset -fx one 2>/dev/null
cat logs/or_grid_jobs.txt | while read -r line; do
  while (( $(jobs -r | wc -l) >= 6 )); do sleep 2; done
  one "$line" &
done
wait
echo "grid done $(date -u +%H:%M:%S)"
