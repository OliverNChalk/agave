#!/usr/bin/env bash

here=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

set -o errexit
set -o nounset

# Testing remotes functionality.
#
# `upstream/master` is included in the synchronization logic, and the pre-rebase
# hook would not able to do it's work correctly without it.  It should handle
# absence of the `upstream/master` branch gracefully.

# shellcheck source=scripts/pre-rebase-hook/test-common
source "$here/test-common"
# shellcheck source=scripts/pre-rebase-hook/test-setups
source "$here/test-setups"

setupSandbox

setup1 upstream 2023-06-28 "git@mock-github.com:solana/invalidator.git"

git branch --quiet --remotes --delete upstream/master

# As `upstream/master` was removed, we need to provide an explicit upstream for
# the rebase operation.
runGitRebase master

assertExitCode 128
assertStdout ''
# We do want single quotes here.
# shellcheck disable=SC2016
assertStderr 'ERROR:
    Could not find a remote that holds synchronization branches needed to track
    the "invalidator" repository `master` branch rebases.  Can not make sure
    your rebase is safe.

    A remote needs to contain the following branches:

    master
    sync/master-upstream
    sync/master/upstream/*
    sync/master-local
    sync/master/local/*

    You can add an upstream pointing at the invalidator repo like this:

    git remote add upstream git@github.com:solana/invalidator.git
    git fetch upstream

    Or you can remove the rebase hook, to disable checking altogether:

    rm .git/hook/pre-rebase
fatal: The pre-rebase hook refused to rebase.'
