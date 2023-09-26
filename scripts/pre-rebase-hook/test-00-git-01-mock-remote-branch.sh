#!/usr/bin/env bash

here=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

set -o errexit
set -o nounset

# Tests `gitMockRemoteBranch` helper.
#
# `gitMockRemoteBranch` creates a remote tracking branch somewhat manually,
# bypassing normal tooling.  Make sure it still works, in case anything changes.

# shellcheck source=scripts/pre-rebase-hook/test-common
source "$here/test-common"

setupSandbox

git remote add mockUpstream "git@mock-github.com:org/repo.git"

cat >file.txt <<EOF
Content does not matter
EOF
git add file.txt
git commit --quiet --message 'First commit'

cat >>file.txt <<EOF

More text
EOF
git add file.txt
git commit --quiet --message 'Second commit'

gitMockRemoteBranch mockUpstream path/branch1 @~
gitMockRemoteBranch mockUpstream another/path/branch2 HEAD

git branch --list --remotes --no-column --no-abbrev --verbose >actual
cat >expected <<EOF
  mockUpstream/another/path/branch2 0406569db18347e30ddbb63621a9a2bd3cf951ed Second commit
  mockUpstream/path/branch1         8b339940dc49af50e62661047618c5a417367b19 First commit
EOF

diff -u actual expected
