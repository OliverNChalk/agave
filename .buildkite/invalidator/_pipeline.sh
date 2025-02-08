#!/usr/bin/env bash

set -e

set_affected_files() {
  local pr_number=$1

  if [[ $# -ne 1 ]]; then
    echo "ERROR: set_affected_files takes 1 argument"
    exit 1
  fi

  local pr_files

  local prev_ops
  local last_exit
  prev_ops=$(set +o)
  set +o errexit

  pr_files=$(gh pr diff --name-only "$pr_number" 2>&1)
  last_exit=$?

  eval "$prev_ops"

  # `gh pr diff` fails if the number of changed files is more than 300 or if the
  # diff is too big.  In either case, running all steps seems an OK workaround
  # for now.
  if [[ "$last_exit" -eq 1 \
    && "$pr_files" = *"PullRequest.diff too_large"* ]]; then
    TRIGGER_ALL_STEPS=1
    return
  fi

  if [[ "$last_exit" -ne 0 ]]; then
    cat <<EOM
ERROR: "gh pr diff --name-only $pr_number" failed.
  Error: $last_exit
  Output:
EOM
    printf -- "%s\n" "$pr_files" | sed -e 's/^/  /'
    exit 1
  fi

  readarray -t affected_files <<<"$pr_files"
  if [[ ${#affected_files[*]} -eq 0 ]]; then
    echo "Unable to determine the files affected by this PR"
    exit 1
  fi
}

if [[ -z "$TRIGGER_ALL_STEPS" ]]; then
  set_affected_files "$BUILDKITE_PULL_REQUEST"
fi

# `set_affected_files` above may set $TRIGGER_ALL_STEPS, so check again.
if [[ -z "$TRIGGER_ALL_STEPS" ]]; then
  echo "~~~ Affected files"
  printf '%s\n' "${affected_files[@]}"
else
  echo "Triggering all steps, ignoring affected files"
fi

declare -a mandatory_affected_files=(
  Cargo.toml$
  Cargo.lock$
  ^ci/buildkite-pipeline.sh
  ^ci/docker/Dockerfile
  ^.buildkite/invalidator/_pipeline.sh
)

affects() {
  if [[ -n "$TRIGGER_ALL_STEPS" ]]; then
    return 0
  fi

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
  local name=$1
  local timeout_in_minutes=$2
  local command=$3
  local retry=${4:-0}

  cat >>pipeline <<EOF
  - name: "$name"
    command: "$command"
    timeout_in_minutes: $timeout_in_minutes
    agents:
      queue: "default"
EOF

  if [[ $retry -gt 0 ]]; then
    cat >>pipeline <<EOF
    retry:
      automatic:
        - limit: $retry
EOF
  fi
}

add_step_parallel() {
  local name=$1
  local timeout_in_minutes=$2
  local parallelism=$3
  local retry=$4
  local command=$5

  cat >>pipeline <<EOF
  - name: "$name"
    command: "$command"
    timeout_in_minutes: $timeout_in_minutes
    agents:
      queue: "default"
    parallelism: $parallelism
    retry:
      automatic:
        - limit: $retry
EOF
}

add_step_in_docker() {
  local name=$1
  local timeout_in_minutes=$2
  local command=$3
  local retry=${4:-0}

  command="ci/docker-run-default-image.sh ${command}"

  add_step "$name" "$timeout_in_minutes" "$command" "$retry"
}

add_step_in_docker_parallel() {
  local name=$1
  local timeout_in_minutes=$2
  local parallelism=$3
  local retry=$4
  local command=$5

  command="ci/docker-run-default-image.sh ${command}"

  add_step_parallel \
    "$name" "$timeout_in_minutes" "$parallelism" "$retry" "$command"
}

wait_step() {
  echo "  - wait" >>pipeline
}

echo "steps:" >pipeline

## sanity
add_step sanity 5 \
  ci/test-sanity.sh

## shellcheck
if affects .sh$ ; then
  add_step shellcheck 5 \
    ci/shellcheck.sh
fi

## pre-rebase hook
if affects ^scripts/pre-rebase-hook.sh ^scripts/pre-rebase-hook/ ; then
  add_step pre-rebase-hook 5 \
    ci/test-pre-rebase-hook.sh
fi

## checks
add_step_in_docker checks 20 \
  ci/test-checks.sh
wait_step

## stable
if affects .rs$ ^ci/rust-version.sh ^ci/test-stable.sh ; then
  parallelism=2
  for i in $(seq 1 $parallelism); do
    add_step_in_docker "stable-partition-$i" 60 \
      "ci/stable/run-partition.sh $i $parallelism" 3
  done

  add_step_in_docker localnet 30 \
    ci/stable/run-localnet.sh
fi

## docs
if affects .rs$ ^ci/rust-version.sh ^ci/test-docs.sh ; then
  add_step doctest 15 \
    ci/test-docs.sh
fi

if affects .rs$ ; then

  ## local-cluster
  parallelism=6
  for i in $(seq 1 $parallelism); do
    add_step_in_docker "local-cluster-partition-$i" 60 \
      "ci/stable/run-local-cluster-partially.sh $i $parallelism" 3
  done

fi

cat pipeline

# shellcheck disable=SC2094
buildkite-agent pipeline upload <pipeline
