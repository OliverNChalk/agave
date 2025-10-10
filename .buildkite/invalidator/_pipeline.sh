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

# On parallelism
#
# The following is the reasoning as to why `parallelism` parameter is set to
# 8/16 for `stable-partition`/`local-cluster-partition` steps respectively.
#
# Having more steps that are smaller is generally better for scheduling.  And
# better for retries.
#
# In terms of scheduling, as he have 9 agents, and the steps have rather varying
# length, some agents before free much faster then the rest, and if they do not
# have any work to do, we are just wasting an opportunity to do more work
# faster.
#
# If a step fails, CI will retry it up to 3 times.  But we combine multiple
# tests in the same CI step.  Meaning we are going to retry tests that have
# actually succeeded.  Smaller steps mean that our retries waste less compute
# and finish faster.
#
# When split into 2, `stable-partition` steps can run for about 15 minutes, but
# can take up to 40 minutes due to internal retries - this is when steps
# complete successfully still on the first attempt.
# When when split into 6, `local-cluster-partition` steps can take between 8 and
# 17 minutes.  But with retries, time can go up.
#
# With 2 splits for `stable-partition` and 6 splits for
# `local-cluster-partition` we see that the first agent becomes vacant after
# less than 9 minutes of the pipeline execution.  Yet, the pipeline still takes
# 40 minutes due to the length of the longest task.  For the last 20 minutes,
# only 3 agents have any work to do.  And for the last 12 minutes, only a single
# agent is busy.
#
# There is, unfortunately, considerable overhead in scheduling and compilation
# on each agent, for every step.  Seems like we do not share compilation results
# from the previous invocations even on the same agent.
#
# `stable-partition` tests take 2 minutes to compile, before any execution can
# start.  `local-cluster-partition` takes take 3 minutes to compile.  There are
# also scheduling delays in the order of 1/2 minute, between two tasks on the
# same agent.
#
# We need to consider this overhead.  And in practice, we see it increasing the
# total pipeline execution time.  With 10 and 30 steps for `stable-partition`
# and `local-cluster-partition` respectively, this overhead is already causing
# the total time to approach 32 minutes.  While a lucky run for the 2/6 splits
# can be just 25 minutes.
#
# 8/16 splits for `stable-partition`/`local-cluster-partition` respectively is
# somewhere in the middle.  It seems like it is still brining the shortest
# execution time up, to around 30 minutes (though I've seen a 22 minute run),
# but it should reduce total time variance and greatly reduce the impact of
# retries, compared to the 2/6 splits case.

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
  parallelism=8
  for i in $(seq 1 $parallelism); do
    add_step_in_docker "stable-partition-$i" 60 \
      "ci/stable/run-partition.sh $i $parallelism" 3
  done

  add_step_in_docker localnet 30 \
    ci/stable/run-localnet.sh
fi

## docs
if affects .rs$ ^ci/rust-version.sh ^ci/test-docs.sh ; then
  # run in container due to libclang-dev dependencies for rocksdb.
  add_step_in_docker doctest 15 \
    ci/test-docs.sh
fi

if affects .rs$ ; then

  ## local-cluster
  parallelism=16
  for i in $(seq 1 $parallelism); do
    add_step_in_docker "local-cluster-partition-$i" 60 \
      "ci/stable/run-local-cluster-partially.sh $i $parallelism" 3
  done

fi

cat pipeline

# shellcheck disable=SC2094
buildkite-agent pipeline upload <pipeline
