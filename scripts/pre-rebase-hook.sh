#!/usr/bin/env bash

# ================================ IMPORTANT ==================================
#
#   Increment value of the `version` variable below every time you update this
#   script.
#
# =============================================================================

set -o errexit
set -o nounset

declare args
args=( "$@" )
declare me
me=$( realpath "${BASH_SOURCE[0]}" )

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

# Checks if there is a newer version of this hook forwarder in the repo.  And if
# so, replace itself with it.
self_update() {
  local me=$1
  local myVersion=$2
  local repoLocation=$3
  shift 3

  local -a args
  args=( "$@" )

  if [[ ! -r "$repoLocation" ]]; then
    # Avoid blocking the rebase command on branches with no hook.
    return
  fi

  local repoVersion
  repoVersion=$(
    sed --silent --expression='
        s@^declare version=\([0-9]\+\)$@\1@p
      ' "$repoLocation" 2>/dev/null
    ) || true

  if [[ -z "$repoVersion" || "$myVersion" -ge "$repoVersion" ]]; then
    return
  fi

  if ! cp "$repoLocation" "$me"; then
    cat <<EOF >&2
WARNING:
  Failed while updating the rebase hook.
  Installed version: $myVersion
  Repository version: $repoVersion
  Repository hook location: $repoLocation

  Consider fixing it manually:

cp "$repoLocation" "$me"
EOF
    return
  fi

  exec "$me" "${args[@]}"
}

# Runs hook implementation from the current workspace.
forward() {
  local -a args
  args=( "$@" )

  # Avoid completely blocking the rebase command if something is wrong.
  if [[ ! -e "$hookScript" ]]; then
    cat <<EOF >&2
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
}

# === Main sequence ===

# This script can update itself, should it find a newer version in the current
# workspace when executed.
#
# "self_update()" uses sed to look for this version line.  Keep the format.
declare version=1
declare meInRepo=scripts/pre-rebase-hook.sh

# Allow execution from both `.git/hook` as well as from `scripts/`.

declare topLevel
topLevel=$( git rev-parse --show-toplevel )
declare hookScript
hookScript=$( realpath "$topLevel/scripts/pre-rebase-hook/run.sh" )

self_update "$me" "$version" "$topLevel/$meInRepo" \
  "${args[@]}"

forward "${args[@]}"
