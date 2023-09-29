#!/usr/bin/env bash

here=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

set -o errexit
set -o nounset

# Testing remotes functionality.
#
# With a single remote configured that contains a set of branches for tracking
# `master` branch status the hook should silently exit.

# shellcheck source=scripts/pre-rebase-hook/test-common
source "$here/test-common"
# shellcheck source=scripts/pre-rebase-hook/test-setups
source "$here/test-setups"

setupSandbox

setup1 upstream 2023-06-28 "git@mock-github.com:solana/invalidator.git"

# `user1` is expected to be configured to track `upstream/master`, so a rebase
# with no arguments should just rebase it on top of `master`.
runGitRebase

assertSuccesfulRebase 'First, rewinding head to replay your work on top of it...
Applying: U1: Extended a.txt
Applying: U2: Extended b.txt' \
  'Rebasing (1/2)
Rebasing (2/2)
Successfully rebased and updated refs/heads/user1.'
assertExitCode 0
