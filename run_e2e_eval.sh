#!/usr/bin/env bash
# End-to-end R2R evaluation launcher (same usage as VLN-CE-master's
# run_final_method_eval.sh):
#
#   bash run_e2e_eval.sh 7                       # one OpenNav episode id
#   bash run_e2e_eval.sh 7,11,13 --workers 2     # explicit ids, 2 parallel shards
#   bash run_e2e_eval.sh --all --workers 2       # all 100 OpenNav ids (official val_unseen poses)
#   bash run_e2e_eval.sh --episode-indices 0,3,6,9,18,27,45,126,204,219
#   bash run_e2e_eval.sh --dry-run 7             # preflight only, no GPU/Habitat
#   bash run_e2e_eval.sh --list                  # print the 100 ids
#   bash run_e2e_eval.sh --resume outputs/e2e_eval/<round_dir>
#
# Unknown options are forwarded to scripts/evaluate_point_navigation.py
# (e.g. --views 6 --vlm-timeout 240).  Smoke without VLM cost:
#   bash run_e2e_eval.sh 7 --vlm-backend heuristic --targets 1 \
#     --max-steps-per-target 12 --sequence-max-exploration-hops 1 --run-tag smoke
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -f local_env.sh ]]; then
  echo "local_env.sh is missing; copy local_env.sh.example and fill it in" >&2
  exit 2
fi
# shellcheck disable=SC1091
source local_env.sh
exec python scripts/run_end_to_end_eval.py "$@"
