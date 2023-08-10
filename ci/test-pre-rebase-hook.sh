#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

source ci/_

# Run tests for the pre-rebase hook.  Only used in the "invalidator" repo.
_ scripts/pre-rebase-hook/test.sh
