#!/usr/bin/env bash

here=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
args=( "$@" )

# This hook helps deal with the `master` branch rebases, that happen when
# changes from the upstream repo are integrated into the `master` branch.
#
# To install, copy it into your local `.git/hooks/` folder:
#
#   cp scripts/pre-rebase-hook.sh .git/hooks/pre-rebase
#
# ---
#
# As the actual hook may change, we do not want to copy it into the local
# workspace.  Instead, always use the latest version in the repo.
#
# A small downside is that going to an older commit in the repo will also
# downgrade the hook.  But it seems to be the lesser evil, that, hopefully, will
# not really bother anyone.
#
# Allow execution both from `.git/hook` as well as from `scripts/`.

if [[ "$here" =~ /scripts$ ]]; then
  hookScript=$here/pre-rebase-hook/run.sh
else
  # `git rebase` will invoke the hook from the workspace top level directory.
  hookScript=$here/scripts/pre-rebase-hook/run.sh
fi

# Avoid completely blocking the rebase command if something is wrong.
if [[ ! -e "$hookScript" ]]; then
  cat <<EOF
WARNING:
    Failed to find \`pre-rebase\` hook script implementation.
    Missing: $hookScript
    Allowing rebase without any checks.

    Fix it or remove \`.git/hook/pre-rebase\` to avoid this message.
EOF
  exit 0
fi

# shellcheck source=scripts/pre-rebase-hook/run.sh
exec "$hookScript" "${args[@]}"
