# shellcheck disable=SC2148
#
# Invalidator attack scenarios. This is expected to be sourced by the
# continuous-test.sh script.
#

set -o errexit
set -o nounset
set -o pipefail

for requiredVar in commonArgs repairShArgs runtime sleeptime invalidatorClient; do
  if ! declare -p "$requiredVar" >/dev/null; then
    cat <<EOM
invalidator-tests expects $requiredVar to be set.  Functions defined in this
file are defined with an assumption that this variable is a global variable
defined by the script that includes this script via the 'source' command.
EOM
    exit 1
  fi
done

attack_invalidateLeaderBlock() {
  local kind=$1
  shift
  [[ $# -ne 0 ]] && die "attack_invalidateLeaderBlock takes 1 argument"

  # shellcheck disable=SC2154
  "$invalidatorClient" "${commonArgs[@]}" \
    configure-invalidate-leader-block --invalidation-kind "$kind"
  # shellcheck disable=SC2154
  sleep "$runtime"
  "$invalidatorClient" "${commonArgs[@]}" \
    configure-invalidate-leader-block
  # shellcheck disable=SC2154
  sleep "$sleeptime"
}

attack_dropTurbineVotes() {
  "$invalidatorClient" "${commonArgs[@]}" \
    configure-drop-turbine-votes --drop true
  sleep "$runtime"
  "$invalidatorClient" "${commonArgs[@]}" \
    configure-drop-turbine-votes --drop false
  sleep "$sleeptime"
}

attack_repairTests() {
  local test=$1
  shift
  local -a extraConfig=( "$@" )

  # shellcheck disable=SC2154
  "${here}/repair-tests.sh" "${repairShArgs[@]}" --invalidator-client "$invalidatorClient" --test "$test" \
    "${extraConfig[@]}"
  sleep "$runtime"
  "${here}/repair-tests.sh" "${repairShArgs[@]}" --invalidator-client "$invalidatorClient" --test disable
  sleep "$sleeptime"
}

attack_gossipPacketFlood() {
  local strategy=$1
  shift
  local -a extraConfig=( "$@" )

  "$invalidatorClient" "${commonArgs[@]}" \
    configure-gossip-packet-flood --flood-strategy "$strategy" \
    "${extraConfig[@]}"
  sleep "$runtime"
  "$invalidatorClient" "${commonArgs[@]}" \
    configure-gossip-packet-flood
  sleep "$sleeptime"
}

attack_replayStage() {
  local attack=$1
  shift
  [[ $# -ne 0 ]] && die "attack_replayStage takes 1 argument"

  "$invalidatorClient" "${commonArgs[@]}" \
    configure-replay-stage-attack --selected-attack "$attack"
  sleep "$runtime"
  "$invalidatorClient" "${commonArgs[@]}" \
    configure-replay-stage-attack
  sleep "$sleeptime"
}

attack_delayBroadcast() {
  "$invalidatorClient" "${commonArgs[@]}" configure-send-duplicate-blocks \
    --turbine-send-delay-ms 1600
  sleep "$runtime"
  "$invalidatorClient" "${commonArgs[@]}" configure-send-duplicate-blocks
  sleep "$sleeptime"
}

attack_sendDuplicateBlocks() {
  "$invalidatorClient" "${commonArgs[@]}" configure-send-duplicate-blocks \
    --new-entry-index-from-end 0 \
    --num-duplicate-validators 1 \
    --send-original-after-ms 0
  sleep "$runtime"
  "$invalidatorClient" "${commonArgs[@]}" configure-send-duplicate-blocks
  sleep "$sleeptime"
}

attack_sendDuplicateLeafNodes() {
  "$invalidatorClient" "${commonArgs[@]}" configure-send-duplicate-blocks \
    --new-entry-index-from-end 2 \
    --send-original-after-ms 0 \
    --leaf_node_partitions 2
  sleep "$runtime"
  "$invalidatorClient" "${commonArgs[@]}" configure-send-duplicate-blocks
  sleep "$sleeptime"
}
