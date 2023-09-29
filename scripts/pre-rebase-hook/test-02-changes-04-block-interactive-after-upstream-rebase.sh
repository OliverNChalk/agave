#!/usr/bin/env bash

here=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

set -o errexit
set -o nounset

# Test rebase functionality.
#
# A rebase of a user branch when the `master` branch was just rebased is the
# main target case of the pre-rebase hook.  The hook is invoked for an
# interactive rebase as well, and so it will block a rebase attempt that
# includes commits that have already been rebase.  Make sure the error message
# shown is correct.

# shellcheck source=scripts/pre-rebase-hook/test-common
source "$here/test-common"
# shellcheck source=scripts/pre-rebase-hook/test-setups
source "$here/test-setups"

setupSandbox

setup2 upstream 2023-05-17 2023-06-28 \
  "git@mock-github.com:solana/invalidator.git"

# `user1` is expected to be configured to track `upstream/master`, so a rebase
# with no arguments should just rebase it on top of `master`.
GIT_EDITOR=true \
  runGitRebase --interactive

assertStdout ''
# We do want single quotes here.
# shellcheck disable=SC2016
assertStderr 'ERROR:
  You are trying to rebase the following changes that are already part of the
  "upstream/master" branch:

a20bd5ea8a79a0816ca7b9a271e39c2e7c163414 I3: Removed from c.txt
05750cf24759027fbbce5e55751499815c67e1fc I2'\'': Inserted into c.txt
15f08b6278e9c9baa4080211009534067f608945 I1'\'': Add c.txt

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
assertExitCode 128

# A command that the hook printed should work with no issues.
GIT_EDITOR=true \
  runGitRebase --interactive \
  --onto=upstream/master upstream/sync/master/local/2023-06-28

assertStdout ''
assertStderr 'Rebasing (1/2)
Rebasing (2/2)
Successfully rebased and updated refs/heads/user1.'
assertExitCode 0

# Make sure produced history matches the expectations.
runGitLog --oneline
assertStdout "a320039 U2: Extended b.txt
ad226a3 U1: Extended a.txt
0422c2a I4': Added d.txt
ddd463c I3': Removed from c.txt
fdd9293 S3: Modified c.txt
8857f05 I2'': Inserted into c.txt
9ab8cd7 I1'': Add c.txt
7165395 S2: Add b.txt
504298d S1: Add a.txt
ff64ae0 S0: Add readme.txt"
assertStderr ''
assertExitCode 0
