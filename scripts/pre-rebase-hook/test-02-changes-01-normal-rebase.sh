#!/usr/bin/env bash

here=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

set -o errexit
set -o nounset

# Test rebase functionality.
#
# The most common case of a rebase is the one that does not involve any history
# that is part of the changes merged upstream.  The hook must not show any
# messages and allow the rebase to go though.

# shellcheck source=scripts/pre-rebase-hook/test-common
source "$here/test-common"
# shellcheck source=scripts/pre-rebase-hook/test-setups
source "$here/test-setups"

setupSandbox

setup1 upstream 2023-06-28 "git@mock-github.com:solana/invalidator.git"

# "d.txt" is only added in "I4" which is part of `master`, but is not part of
# `user1` just yet.
assertFileAbsent d.txt

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

# And now we should see "d.txt" in all its glory.
assertFileContent d.txt 'Now with d.txt'
