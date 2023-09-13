#!/usr/bin/env bash

# Script that runs upstream synchronization process with the solana repository.
# Details are in ../docs/upstream-sync.md.

set -o errexit
set -o nounset
set -o pipefail

here=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
cargo=
customCargo=

# = Arguments =

date=
preCheck=
runTests=
branch=master

usage() {
  cat <<EOM
Usage:
  $0
     [--branch <branch>]
     --date <ISO-sync-date>
     [--pre-check]
     [--run-tests]
     [--cargo <path>]

Runs a step-by-step rebase of the \`invalidator/<branch>\` on top of the latest
changes pulled from \`solana/<branch>\`.

You should be on a branch called \`<branch>-next\` when you invoke this script.

Arguments:
  --branch <branch>  Branch name that should be rebased.
      Optional.  Default: master

  --date  Specifies date of the synchronization.  Script will take changes from
      \`solana/<branch>\` up to, but not including this date.
      Required.

  --pre-check  Runs compilation, formatting and other checks before any rebase
      operations.  Useful when resuing a rebase, after making adjustements.
      Makes sure that the next rebase will happen starting from a clean state.
      Optional.  Default: no

  --run-tests  Runs \`cargo test\` in addition to the normal compilation and
      other verifications, after every full rebase.  It will probably be too
      slow for any practical purposes.
      Optional.  Default: no

  --cargo  Specifies location of the \`cargo\` script that is normally located
      in the workspace root.
      Optional.  Default: ../cargo
EOM
}

# = Argument processing =

die() {
  printf 'ERROR: %s' "$1" >&2
  shift
  [[ $# -gt 0 ]] && printf ' %s' "$@" >&2
  echo >&2

  exit 1
}

while [[ $# -gt 0 ]]; do
  name=$1
  shift

  case $name in
    --branch)
      [[ $# -eq 0 ]] && die '"--branch" requires a branch name argument'
      branch=$1
      shift
      ;;
    --date)
      [[ $# -eq 0 ]] && die '"--date" requires an ISO date argument'
      date=$1
      shift
      [[ ! "$date" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] \
        && die '"--date" must match YYYY-mm-dd, got:' "\"$date\""
      ;;
    --pre-check)
      preCheck=y
      ;;
    --run-tests)
      runTests=y
      ;;
    --cargo)
      [[ $# -eq 0 ]] && die '"--cargo" requires a path argument'
      cargo=$1
      customCargo=y
      shift
      ;;
    -h|-\?|--help)
      usage
      exit
      ;;
    *)
      printf 'ERROR: Unexpected argument: "%s"\n\n' "$1" >&2
      usage
      exit 2
      ;;
  esac
done

[[ -z "$date" ]] && {
  echo 'ERROR: "--date" is required' >&2
  usage
  exit 2
}

# = Implementation =

find_cargo() {
  if [[ -z "$cargo" ]]; then
    cargo=$(readlink -f "${here}/../cargo")

    if [[ -z "$cargo" ]]; then
      cat >&2 <<EOM
ERROR: Failed to find cargo.
On MacOS readlink doesn't support -f.
Consider switching to gnu readlink with 'brew install coreutils' and then
symlink readlink as /usr/local/bin/readlink.
EOM
      exit 3
    fi
  else
    # Make sure we use the specified cargo command, even when we run it from a
    # different directory.  Such as programs/sbf.
    cargo=$( readlink -f "$cargo" )
  fi

  if [[ ! -x "$cargo" ]]; then
    cat >&2 <<EOM
ERROR: Expected a 'cargo' script at "$cargo"
EOM
    exit 3
  fi
}

print_restart_command() {
  local extraArgs=
  [[ $# -gt 0 ]] && extraArgs=$1

  echo -n "$0 --date \"$date\""
  [[ -n "$customCargo" ]] && echo -n " --cargo \"$cargo\""
  [[ -n "$extraArgs" ]] && echo -n " $extraArgs"
  [[ -n "$runTests" ]] && echo -n " --run-tests"
  echo
}

run_one_rebase() {
  local nextBase
  nextBase=$(
      git rev-list "${branch}-next..sync/${branch}-upstream" | tail -n 1
    ) \
      || die "'git rev-list $branch-next..sync/$branch-upstream' failed"

  if [[ -z "$nextBase" ]]; then
    echo "=== Rebase is done ==="
    exit
  fi

  local mergeBase
  mergeBase=$( git merge-base "${branch}-next" "sync/${branch}-upstream" ) \
    || die "'git merge-base $branch-next sync/$branch-upstream' failed"

  if ! git rebase --onto "$nextBase" "$mergeBase"; then
    cat >&2 <<'EOM'
Failed to rebase.
  Do:
    1. Finish branch rebase:

git mergetool
git add -u
git rebase --continue

    2. If necessary, repeat step 1 until the whole branch is rebased.

    3. Restart the outer rebase script by running:

EOM
    print_restart_command --pre-check
    exit 1
  fi
}

run_cargo_check() {
  local where=.
  [[ $# -gt 0 ]] && where=$1

  if ! ( cd "$where" && "$cargo" check --tests ); then
    if [[ "$where" = "." ]]; then
      echo "Failed: $cargo check --tests"
    else
      echo "Failed: cd \"$where\" && $cargo check --tests"
    fi
    cat >&2 <<EOM
  Do:
    1. Fix compilation errors.
    2. Add changes into a commit that will be dissoled later:

git add -u
git commit --message "DO NOT SUBMIT: Fixup for \\"\$(
        git show --oneline --no-patch "\$( \\
            git merge-base "sync/${branch}-upstream" HEAD \\
        )" \\
    )\\""

    3. Restart the rebase by running:

EOM
    print_restart_command --pre-check
    exit 1
  fi
}

run_fmt() {
  local where=.
  [[ $# -gt 0 ]] && where=$1

  ( cd "$where" && "$cargo" nightly fmt ) \
    || die "'cd \"$where\" && $cargo nightly fmt' failed"

  local untracked
  untracked=$( cd "$where" && git status --untracked-files=no --porcelain ) \
    || die "'cd \"$where\" && git status --untracked-files=no --porcelain' failed"

  if [[ -n "$untracked" ]]; then
    if [[ "$where" = "." ]]; then
      echo "Failed: Wrong code formatting or Cargo.lock was updated"
    else
      echo "Failed: Wrong code formatting or Cargo.lock was updated in $where"
    fi
    cat >&2 <<EOM
  Do:
    1. If formatting is fine, add changes into a commit that will be dissoled
       later:

git add -u
git commit --message "DO NOT SUBMIT: Fixup for \\"\$(
        git show --oneline --no-patch "\$( \\
            git merge-base "sync/${branch}-upstream" HEAD \\
        )" \\
    )\\""

    2. Restart the rebase by running:

EOM
    print_restart_command --pre-check
    exit 1
  fi
}

run_tests() {
  if ! cargo test; then
    echo "Failed: $cargo test"
    cat >&2 <<EOM
  Do:
    1. Fix tests.
    2. Update the last commit with

git add -u
git commit --message "DO NOT SUBMIT: Fixup for \\"\$(
        git show --oneline --no-patch "\$( \\
            git merge-base "sync/${branch}-upstream" HEAD \\
        )" \\
    )\\""

    3. Restart the rebase by running:

EOM
    print_restart_command --pre-check
    exit 1
  fi
}

run_all_checks() {
  run_cargo_check
  run_fmt
  run_cargo_check programs/sbf
  run_fmt programs/sbf
  if [[ -n "$runTests" ]]; then
    run_tests
  fi
}

# = Execution =

find_cargo

currentBranch=$( git branch --show-current )
if [[ "$currentBranch" != "${branch}-next" ]]; then
  cat >&2 <<EOM
Failed:
  slow-rebase.sh is expects "${branch}-next" to be the current branch.
  Currently checked out branch: "$currentBranch"

  Switch to "${branch}-next" with:

git switch ${branch}-next
EOM
  exit 1
fi

echo -n "Changes left: "
git rev-list "${branch}-next..sync/${branch}-upstream" | wc -l

[[ -n "$preCheck" ]] && run_all_checks

while true; do
  run_one_rebase
  run_all_checks
done

# vim: ft=bash sw=2 sts=2 ts=2 et:
# -*- mode: bash -*-
# code: language=bash
