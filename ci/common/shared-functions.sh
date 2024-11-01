#!/usr/bin/env bash

need_to_upload_test_result() {
  local branches=(
    "$EDGE_CHANNEL"
    "$BETA_CHANNEL"
    "$STABLE_CHANNEL"
  )

  for n in "${branches[@]}"; do
    if [[ "$CI_BRANCH" == "$n" ]]; then
      return 0
    fi
  done

  # Invalidator `master` branch is rebased regularly, and we want to collect
  # stats for CI runs for those rebases.  The rebase process uses `master-next`
  # as a candidate branch for the next `master` content.
  #
  # Candidates are created as normal PRs, though they are never merged, they
  # are only used for CI runs.  So we are looking for pull requests for the
  # `master` branch, with the source branch named `master-next`, coming from any
  # fork.  "$CI_BRANCH" will have a value of "user-fork:master-next", so we
  # pattern match.
  if [[ -n "$CI_PULL_REQUEST" \
      && "$CI_BASE_BRANCH" == "master" \
      && "$CI_BRANCH" == *:master-next ]]; then
    return 0
  fi

  return 1
}

exit_if_error() {
  if [[ "$1" -ne 0 ]]; then
    exit "$1"
  fi
}
