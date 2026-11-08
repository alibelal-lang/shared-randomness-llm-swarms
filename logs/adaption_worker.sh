#!/bin/zsh
# One Adaption job at a time (the rate limit is per user). Jobs are lines in logs/adaption_queue.txt,
# taken from the top; append lines to add work, a line "END" stops the worker.
cd "/Users/alibelal/Swarm alignment"
export ADAPTION_MAX_RATE=${ADAPTION_MAX_RATE:-8}
Q=logs/adaption_queue.txt
touch $Q
while true; do
  line=$(head -n1 $Q)
  if [[ -z "$line" ]]; then sleep 20; continue; fi
  tail -n +2 $Q > $Q.tmp && cat $Q.tmp > $Q && rm -f $Q.tmp
  [[ "$line" == "END" ]] && { echo "[worker] END $(date -u +%H:%M:%S)"; exit 0; }
  echo "[worker] START $(date -u +%H:%M:%S) $line"
  eval "$line"
  echo "[worker] DONE rc=$? $(date -u +%H:%M:%S)"
done
