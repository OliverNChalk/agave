#!/usr/bin/env bash

here=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

set -e

# Testing remotes functionality.
#
# When there are two remotes that both contain branches for tracking the
# `master` rebases and no hook specific configuration exist, the hook must show
# a message and exit.

# shellcheck source=scripts/pre-rebase-hook/test-common
source "$here/test-common"
# shellcheck source=scripts/pre-rebase-hook/test-setups
source "$here/test-setups"

setupSandbox

setup1 upstream1 2023-06-28 "git@mock-github.com:solana/invalidator.git"

git remote add upstream2 "git@mock-github.com:solana/invalidator2.git"
createRebaseTrackingBranchesFor upstream2 2023-05-17 \
  "master^{/^S1:}" "master^{/^I1':}" "master^{/^I3:}"

# `user1` is expected to be configured to track `upstream/master`, so a rebase
# with no arguments should just rebase it on top of `master`.
runGitRebase

assertExitCode 128
assertStdout ''
assertStderr 'ERROR:
    Could not determine a unique upstream remote for the "invalidator" repo.
    At least two candidates found: "upstream1" and "upstream2"
    Remove one of these remotes or select the desired one with

    git config --local invalidator-repo.upstream "upstream1"

    or

    git config --local invalidator-repo.upstream "upstream2"
fatal: The pre-rebase hook refused to rebase.'
