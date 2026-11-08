#!/bin/zsh
# Chain: wait for the first sweep pass -> second pass for files that arrived later -> exploration summary ->
# N up to 64 and the coupling dose-response on the original Adaption policies.
cd "/Users/alibelal/Swarm alignment"
while pgrep -f "swarm.sweep explore" > /dev/null; do sleep 20; done
echo "pass 1 finished $(date -u +%H:%M:%S)"
.venv/bin/python -u -m swarm.sweep explore 2>&1 | tail -3
.venv/bin/python -m swarm.explore_report > results/explore/summary.txt 2>&1
echo "grid.csv: exploration summary written $(date -u +%H:%M:%S)"
for m in qwen3.8-27b glm-5.2 qwen3.5-397b gemma-4-31b-it gemma-4-26b-a4b-it; do
  NOBOOT=1 logs/run_confirm3.sh policies/adaption_$m.json results/ext/orig_$m f0m0 f0m1 f1m0 f1m1
  echo "orig ext done $m $(date -u +%H:%M:%S)"
done
