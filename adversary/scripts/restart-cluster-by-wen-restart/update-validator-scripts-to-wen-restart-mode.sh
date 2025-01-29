#!/usr/bin/env bash

set -o errexit

wen_restart_path="/home/sol/invalidator/wen-restart"
rm "$wen_restart_path" &> /dev/null || true

scripts=(
  "/home/sol/invalidator/scripts/validator.sh"
  "/home/sol/invalidator/scripts/bootstrap-validator.sh"
)

for script_path in "${scripts[@]}"; do
  sed -i '/wait-for-supermajority/d' "$script_path"
  sed -i '/expected-shred-version/d' "$script_path"
  sed -i '/expected-bank-hash/d' "$script_path"
  sed -i '/wen-restart/d' "$script_path"

  sed -i "/ARGS=/a\  --wen-restart $wen_restart_path \\\\" "$script_path"
  # shellcheck disable=SC1003
  sed -i '/ARGS=/a\  --wen-restart-coordinator tiv1zkpDdumabxfZisVjuQgDzGcVSVDTEaHJD6ueVuK \\' "$script_path"
done
