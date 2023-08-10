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

# Explicitly select an upstream we want to use.
git config --local invalidator-repo.upstream upstream2

# `user1` is expected to be configured to track `upstream/master`, so a rebase
# with no arguments should just rebase it on top of `master`.
runGitRebase

assertExitCode 128
assertStdout ''
# We do want single quotes here.
# shellcheck disable=SC2016
assertStderr 'ERROR:
    Upstream remote you have selected in your local workspace configuation does
    not contain the synchronization branches necessary to track "invalidator"
    repository `master` branch rebases.

    Selected remote: upstream2
    Found remote that contains tracking branches: upstream1

    You can remove local configuration:

    git config --local --unset invalidator-repo.upstream

    or update it:

    git config --local invalidator-repo.upstream "upstream1"
fatal: The pre-rebase hook refused to rebase.'
