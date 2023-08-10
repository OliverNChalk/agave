#!/usr/bin/env bash

here=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

set -e

# Runs all the tests for the `pre-rebase` hook scripts.

# 0. Framework tests:

echo "test-00-git-01-mock-remote-branch.sh ..."
"$here/test-00-git-01-mock-remote-branch.sh"
echo "  OK"

# 1. Remotes:
#
# 1.1. With no remotes with the desired set of branches, the hook silently
#   exits.

echo "test-01-remotes-01-no-remotes.sh ..."
"$here/test-01-remotes-01-no-remotes.sh"
echo "  OK"

# 1.2. Hook shows an error and exists when two remotes with the desired set of
#   branches is found, and the configuration does not specify a remote.

echo "test-01-remotes-02-two-matching-no-config.sh ..."
"$here/test-01-remotes-02-two-matching-no-config.sh"
echo "  OK"

# 1.3. Hook correctly selects a single remote, even when two are present, if it
#   is set in the configuration.

echo "test-01-remotes-03-two-matching-with-config.sh ..."
"$here/test-01-remotes-03-two-matching-with-config.sh"
echo "  OK"

# 1.4. When configuration sets a remote, but it is not present, hook exits with
#   an error.

echo "test-01-remotes-04-one-matching-mismatch-config.sh ..."
"$here/test-01-remotes-04-one-matching-mismatch-config.sh"
echo "  OK"

# 1.5. When configuration sets a remote, but it does not contain the desired set
#   of branches, hook exits with an error.

echo "test-01-remotes-05-no-matching-with-config.sh ..."
"$here/test-01-remotes-05-no-matching-with-config.sh"
echo "  OK"

# 1.6. When there is a single remote, the hook allows the rebase.

echo "test-01-remotes-06-single-remote.sh ..."
"$here/test-01-remotes-06-single-remote.sh"
echo "  OK"

# 2. Changes:
#
# 2.1. A rebase that does not contain any of the changes merged upstream should
#   go though with no error.

echo "test-02-changes-01-normal-rebase.sh ..."
"$here/test-02-changes-01-normal-rebase.sh"
echo "  OK"

# 2.2. A rebase that contains changes from the latest upstream state should show
#   a corresponding error and exit with an error code.

echo "test-02-changes-02-block-changes-already-in-master.sh ..."
"$here/test-02-changes-02-block-changes-already-in-master.sh"
echo "  OK"

# 2.3. A rebase that contains changes from an older upstream state should show a
#   corresponding error and exit with an error code.

echo "test-02-changes-03-block-after-upstream-rebase.sh ..."
"$here/test-02-changes-03-block-after-upstream-rebase.sh"
echo "  OK"

# 2.4. An interactive rebase that contains changes from an older upstream state
#   should show a corresponding error and exit with an error code.

echo "test-02-changes-04-block-interactive-after-upstream-rebase.sh ..."
"$here/test-02-changes-04-block-interactive-after-upstream-rebase.sh"
echo "  OK"
