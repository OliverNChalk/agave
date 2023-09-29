#!/usr/bin/env bash

here=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

set -o errexit
set -o nounset

# Test configuration functionality.
#
# When PRE_REBASE_HOOK_NO_CHECK is set, the hook should silently exit, allowing
# any rebase to go though.

# shellcheck source=scripts/pre-rebase-hook/test-common
source "$here/test-common"
# shellcheck source=scripts/pre-rebase-hook/test-setups
source "$here/test-setups"

setupSandbox

setup2 upstream 2023-05-17 2023-06-28 \
  "git@mock-github.com:solana/invalidator.git"

# Run a rebase that includes changes that are already part of `upstream/master`.
# Normally the hook should block this update, but a non-empty
# `PRE_REBASE_HOOK_NO_CHECK` should stop the hook from doing any checks.
#
# `--reapply-cherry-picks` makes git include all commits, even those that seems
# to have been already applied based on the commit message content.

# First check that without `PRE_REBASE_HOOK_NO_CHECK` the rebase is prohibited.
runGitRebase --reapply-cherry-picks \
  --onto upstream/sync/master-upstream "HEAD^{/^I1':}"

assertStdout ''
# We do want single quotes here.
# shellcheck disable=SC2016
assertStderr 'ERROR:
  You are trying to rebase the following changes that are already part of the
  "upstream/master" branch:

a20bd5ea8a79a0816ca7b9a271e39c2e7c163414 I3: Removed from c.txt
05750cf24759027fbbce5e55751499815c67e1fc I2'\'': Inserted into c.txt

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

# Now we want to see a successful rebase, when `PRE_REBASE_HOOK_NO_CHECK` is set
# and the hook is not blocking the command itself.
PRE_REBASE_HOOK_NO_CHECK=yes \
  runGitRebase --reapply-cherry-picks \
  --onto upstream/sync/master-upstream "HEAD^{/^I1':}"

assertSuccesfulRebase "First, rewinding head to replay your work on top of it...
Applying: I2': Inserted into c.txt
Using index info to reconstruct a base tree...
M	c.txt
Falling back to patching base and 3-way merge...
Auto-merging c.txt
Applying: I3: Removed from c.txt
Applying: U1: Extended a.txt
Applying: U2: Extended b.txt" \
  'Rebasing (1/4)
Rebasing (2/4)
Rebasing (3/4)
Rebasing (4/4)
Successfully rebased and updated refs/heads/user1.'
assertExitCode 0
