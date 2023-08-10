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

setup1 upstream 2023-06-28 "git@mock-github.com:solana/invalidator.git"

# Explicitly select an upstream we want to use.
git config --local invalidator-repo.upstream upstream

# It should be enough to remove a single branch, for the remote to be unusable
# for rebase tracking.
git branch --quiet --remotes --delete upstream/sync/master-upstream

# `user1` is expected to be configured to track `upstream/master`, so a rebase
# with no arguments should just rebase it on top of `master`.
runGitRebase

assertExitCode 128
assertStdout ''
assertStderr 'ERROR:
    You have an upstream remote configured as "upstream", but it does not
    contain all the necessary tracking branches.  Remote needs to contain the
    following branches:

    sync/master/upstream/*
    sync/master-upstream
    sync/master/local/*
    sync/master-local

    You can remove the local configuration:

    git config --local --unset invalidator-repo.upstream

    Fix the remote or remove the hook to avoid this message:

    rm .git/hook/pre-rebase
fatal: The pre-rebase hook refused to rebase.'
