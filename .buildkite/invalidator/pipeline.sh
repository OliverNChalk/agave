#!/usr/bin/env bash

set -e

here=$(dirname "$0")

if [[ "$BUILDKITE_PULL_REQUEST" != "false" ]]; then
  "$here"/_pipeline.sh
else
  export TRIGGER_ALL_STEPS=1
  "$here"/_pipeline.sh
fi
