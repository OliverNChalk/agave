#!/usr/bin/env bash

here=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

set -o errexit
set -o nounset

# Runs all the tests for the `pre-rebase` hook scripts.

# Is set to non-empty string if any of the tests fails.
failed=

runGroup() {
  local pattern=$1

  for path in "$here"/$pattern; do
    local name
    name=$( basename "$path" )

    echo "$name ..."

    if [[ ! -x "$path" ]]; then
      echo "  WARNING: Skipping non executable test: $name"
    else
      if ! "$path"; then
        failed=yes
        echo "  FAILED"
      else
        echo "  OK"
      fi
    fi
  done
}

# 0. Framework tests:
runGroup "test-00*.sh"

# 1. Remotes:
runGroup "test-01*.sh"

# 2. Changes:
runGroup "test-02*.sh"

# 3. Configuration:
runGroup "test-03*.sh"

if [ -n "$failed" ]; then
  exit 1
fi
