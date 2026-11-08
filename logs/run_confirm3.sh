#!/bin/zsh
# Amendment 3, E: confirmation sweeps on an Adaption policy, then the layer-2 bootstrap.
# Usage: logs/run_confirm3.sh <policy.json> <out dir> <variant> [variant ...]   e.g. f0m0 f2m3
cd "/Users/alibelal/Swarm alignment"
pol=$1; out=$2; shift 2
SAMPLERS=(independent pair_shared:restore pair_shared:0.3 block:2 pair_shared:0.1 pair_shared:0.2 pair_shared:0.5 pair_shared:0.75 pair_shared:1)
for v in "$@"; do echo $v; done | xargs -P 6 -L 1 zsh -c '.venv/bin/python -m swarm simulate --policy '$pol' --framing ${0[2]} --mapping ${0[4]} --init incumbent_B --seeds 500 --seed-offset 0 --N 4 8 16 32 64 --samplers '"${SAMPLERS[*]}"' --out '$out'/${0}_incumbent_B > /dev/null && echo "sim done $0"'
[[ -n "$NOBOOT" ]] && exit 0
.venv/bin/python -m swarm bootstrap --policy $pol --B 100 --seeds-per 100 --processes 7 --variants "$@" --out $out/bootstrap.json | grep -E "primary|interaction|n_effect_indep |contrast_|B="
