#!/usr/bin/env bash

set -o errexit

: "${BANK_HASH:?Environment variable BANK_HASH is required but not set.}"
: "${SHRED_VERSION:?Environment variable SHRED_VERSION is required but not set.}"
: "${SLOT:?Environment variable SLOT is required but not set.}"

scripts=(
  "/home/sol/invalidator/scripts/validator.sh"
  "/home/sol/invalidator/scripts/bootstrap-validator.sh"
)

for script_path in "${scripts[@]}"; do
  sed -i '/wen-restart/d' "$script_path"
  sed -i '/expected-bank-hash/d' "$script_path"
  sed -i '/expected-shred-version/d' "$script_path"
  sed -i '/wait-for-supermajority/d' "$script_path"

  sed -i "/ARGS=/a\  --expected-bank-hash $BANK_HASH \\\\" "$script_path"
  sed -i "/ARGS=/a\  --expected-shred-version $SHRED_VERSION \\\\" "$script_path"
  sed -i "/ARGS=/a\  --wait-for-supermajority $SLOT \\\\" "$script_path"
done
