#!/usr/bin/env bash

set -e

if [[ -z "$TRIGGER_ALL_STEPS" ]]; then
  # get affected files
  pr_number="$BUILDKITE_PULL_REQUEST"
  readarray -t affected_files < <(gh pr diff --name-only "$pr_number")
  if [[ ${#affected_files[*]} -eq 0 ]]; then
    echo "Unable to determine the files affected by this PR"
    exit 1
  fi
  echo "~~~ Affected files"
  printf '%s\n' "${affected_files[@]}"
else
  echo "trigger all steps so ignore affected files"
fi

affects() {
  if [[ -n "$TRIGGER_ALL_STEPS" ]]; then
    return 0
  fi

  mandatory_affected_files=(
    Cargo.toml$
    Cargo.lock$
    ^ci/buildkite-pipeline.sh
    ^ci/docker/Dockerfile
  )
  for pattern in "${mandatory_affected_files[@]}" "$@"; do
    for file in "${affected_files[@]}"; do
      if [[ $file =~ $pattern ]]; then
        return 0 # affected
      fi
    done
  done

  return 1 # not affected
}

add_step() {
  cat >>pipeline <<EOF
  - name: "$1"
    command: "$2"
    timeout_in_minutes: $3
    cancel_on_build_failing: true
    agents:
      queue: "invalidator"
EOF

  if [[ $retry -gt 0 ]]; then
    cat >>pipeline <<EOF
    retry:
      automatic:
        - limit: $retry
EOF
  fi
}

wait_step() {
  echo "  - wait" >>pipeline
}

echo "steps:" >pipeline

## sanity
add_step "sanity" "ci/test-sanity.sh" 5

## shellcheck
if affects \
  .sh$ \
  ; then
  add_step "shellcheck" "ci/shellcheck.sh" 5
fi

## checks
add_step "checks" "ci/docker-run-default-image.sh ci/test-checks.sh" 20
wait_step

## stable
if affects \
  .rs$ \
  ^ci/rust-version.sh \
  ^ci/test-stable.sh \
  ; then
  add_step "stable" "ci/docker-run-default-image.sh ci/test-stable.sh" 60
fi

## docs
if affects \
  .rs$ \
  ^ci/rust-version.sh \
  ^ci/test-docs.sh \
  ; then
  add_step doctest "ci/test-docs.sh" 15
fi

## local-cluster
if affects \
  .rs$ \
  ; then
  add_step "local-cluster" \
    "ci/docker-run-default-image.sh ci/test-local-cluster.sh" \
    60
fi

## local-cluster-flakey
if affects \
  .rs$ \
  ; then
  add_step "local-cluster-flakey" \
    "ci/docker-run-default-image.sh ci/test-local-cluster-flakey.sh" \
    30
fi

## local-cluster-slow-1
if affects \
  .rs$ \
  ; then
  add_step "local-cluster-slow-1" \
    "ci/docker-run-default-image.sh ci/test-local-cluster-slow-1.sh" \
    60
fi

## local-cluster-slow-2
if affects \
  .rs$ \
  ; then
  add_step "local-cluster-slow-2" \
    "ci/docker-run-default-image.sh ci/test-local-cluster-slow-2.sh" \
    60
fi

cat pipeline

# shellcheck disable=SC2094
buildkite-agent pipeline upload <pipeline
