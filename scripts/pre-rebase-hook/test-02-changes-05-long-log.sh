#!/usr/bin/env bash

here=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

set -o errexit
set -o nounset

# Test to make sure that the log part of the error message does not exceed the
# expected length.

# shellcheck source=scripts/pre-rebase-hook/test-common
source "$here/test-common"
# shellcheck source=scripts/pre-rebase-hook/test-setups
source "$here/test-setups"

setupSandbox

setup2 upstream 2023-05-17 2023-06-28 \
  "git@mock-github.com:solana/invalidator.git"

# Include as many changes in this rebase as possible.  To cause the log message
# to have the maximum size.
runGitRebase --onto upstream/master "remotes/upstream/master^{/^S0:}"

assertExitCode 128
assertStdout ''
# We do want single quotes here.
# shellcheck disable=SC2016
assertStderr 'ERROR:
  You are trying to rebase the following changes that are already part of the
  "upstream/master" branch:

a20bd5ea8a79a0816ca7b9a271e39c2e7c163414 I3: Removed from c.txt
05750cf24759027fbbce5e55751499815c67e1fc I2'\'': Inserted into c.txt
15f08b6278e9c9baa4080211009534067f608945 I1'\'': Add c.txt
716539554911be1521ef160411f9e44dbe55a2d1 S2: Add b.txt
  ... and 1 more commit(s) ...

  Since the last time you rebased your changes on top of the
  "upstream/master" branch, it has been rebased, to include changes from
  the upstream "solana" repo.

  You probably want to rebase your work on top of the new
  "upstream/master" before you continue any work, like this:

  # Rebase, excluding changes already in the "upstream/sync/master/local/2023-06-28" branch.
  git rebase --onto="upstream/master" "upstream/sync/master/local/2023-06-28"

  (If you really know what you are doing, you can skip with check with a
  `--no-verify` argument.)
fatal: The pre-rebase hook refused to rebase.'
