#!/usr/bin/env bash

here=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

set -e

# Test rebase functionality.
#
# While not very likely, an attempt to rebase changes that are already part of
# `upstream/master` or one of the `upstream/sync/master/local/[date]` branches
# is incorrect.  Those changes should not be rebased, except by people
# synchronizing changes from the upstream repo.

# shellcheck source=scripts/pre-rebase-hook/test-common
source "$here/test-common"
# shellcheck source=scripts/pre-rebase-hook/test-setups
source "$here/test-setups"

setupSandbox

setup1 upstream 2023-06-28 "git@mock-github.com:solana/invalidator.git"

# Include changes in the rebase, that are already part of
# `upstream/sync/master-local`.  Normally this should not happen, except when
# the base is selected manually.  But it is still most likely invalid.
runGitRebase --onto upstream/master "remotes/upstream/master^{/^I1':}"

assertExitCode 128
assertStdout ''
# We do want single quotes here.
# shellcheck disable=SC2016
assertStderr 'ERROR:
  You are trying to rebase the following changes that are already part of the
  "upstream/master" branch:

a20bd5ea8a79a0816ca7b9a271e39c2e7c163414 I3: Removed from c.txt
05750cf24759027fbbce5e55751499815c67e1fc I2'\'': Inserted into c.txt

  Consider using using an explicit baseline in your `git rebase` invocation:

  git rebase "upstream/master" "refs/heads/user1"

  (If you really know what you are doing, you can skip with check with a
  `--no-verify` argument.)
fatal: The pre-rebase hook refused to rebase.'
