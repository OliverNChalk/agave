#!/usr/bin/env bash

here=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

set -e

# Testing remotes functionality.
#
# When there are two remotes that both contain branches for tracking the
# `master` rebases and the hook specific configuration exist, the hook allow the
# rebase.

# shellcheck source=scripts/pre-rebase-hook/test-common
source "$here/test-common"
# shellcheck source=scripts/pre-rebase-hook/test-setups
source "$here/test-setups"

setupSandbox

setup1 upstream1 2023-06-28 "git@mock-github.com:solana/invalidator.git"

git remote add upstream2 "git@mock-github.com:solana/invalidator2.git"
createRebaseTrackingBranchesFor upstream2 2023-05-17 \
  "master^{/^S1:}" "master^{/^I1':}" "master^{/^I3:}"

# Explicitly select an upstream we want to use.
git config --local invalidator-repo.upstream upstream1

# `user1` is expected to be configured to track `upstream/master`, so a rebase
# with no arguments should just rebase it on top of `master`.
runGitRebase

assertExitCode 0
assertSuccesfulRebase 'First, rewinding head to replay your work on top of it...
Applying: U1: Extended a.txt
Applying: U2: Extended b.txt' \
  'Rebasing (1/2)
Rebasing (2/2)
Successfully rebased and updated refs/heads/user1.'
