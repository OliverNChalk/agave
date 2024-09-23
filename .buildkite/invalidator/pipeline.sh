#!/usr/bin/env bash

set -e

here=$(dirname "$0")

if [[ "$BUILDKITE_PULL_REQUEST" != "false" ]]; then
  labels=$(gh pr view "$BUILDKITE_PULL_REQUEST" --json labels --jq '.labels[].name')
  for label in $labels; do
    if [[ $label == "noCI" ]]; then
      echo "noCI is present, skip CI"
      exit 0
    fi
  done

  "$here"/_pipeline.sh
else
  export TRIGGER_ALL_STEPS=1
  "$here"/_pipeline.sh
fi
