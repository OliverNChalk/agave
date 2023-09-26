#!/usr/bin/env bash

here=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

set -o errexit
set -o nounset

# Testing remotes functionality.
#
# When there is no remote configured that contains a set of branches for
# tracking `master` branch status the hook should silently exit.

# shellcheck source=scripts/pre-rebase-hook/test-common
source "$here/test-common"
# shellcheck source=scripts/pre-rebase-hook/test-setups
source "$here/test-setups"

setupSandbox

setup1 upstream 2023-06-28 "git@mock-github.com:solana/invalidator.git"

git branch --quiet --remotes --delete upstream/sync/master/upstream/2023-06-28
git branch --quiet --remotes --delete upstream/sync/master-upstream
git branch --quiet --remotes --delete upstream/sync/master/local/2023-06-28
git branch --quiet --remotes --delete upstream/sync/master-local

# `user1` is expected to be configured to track `upstream/master`, so a rebase
# with no arguments should just rebase it on top of `master`.
runGitRebase

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
