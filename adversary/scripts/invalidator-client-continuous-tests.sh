#!/usr/bin/env bash
#
# Loop through invalidator test cases with solana-invalidator-client.
# Ctrl-C to exit.
#

set -o errexit
set -o nounset
set -o pipefail

declare me
me=$( realpath "${BASH_SOURCE[0]}" )
declare here
here=$(realpath "$(dirname "${BASH_SOURCE[0]}")")

#
# === Configuration defaults ===
#

runtime=
sleeptime=
iterations=unbounded
rpcAdversaryKeypair=
attackTarget=
invalidatorClient=solana-invalidator-client

usage() {
  cat <<EOM
Usage:
  $me
    --runtime <secs>
    --sleeptime <secs>
    [--iterations {<num> | unbounded}]
    [--rpc-adversary-keypair <keypair path>]
    [--attack-target {<pubkey> | <IP>}]
    [--invalidator-client <solana-invalidator-client path>]

Configures a running invalidator to run a set of attacks, one after another.
The list of attacks is hardcoded in the scripts.

Arguments:
  --runtime <secs>  Number of seconds to run each attack for.
    Currently, attacks do not specify their duration.  They are executed as long
    as they are active.

  --sleeptime <secs>  Number of seconds to sleep between consequitive attacks.
    The cluster might be affected by an attack for some amount of time after the
    attack has been turned off.  In order to distinguish attack effects more
    clearly in the metrics and logs we wait this long after we disable one
    attack, before we enable the next one.

  --iterations <num>  Number of times to go though the whole attack list.  Use
    "unbounded" to mean, run until stopped with Ctrl+C.
    Optional.  Default: unbounded

  --rpc-adversary-keypair <keypair path>  File holding a private key, that is
    used to sign RPC calls sent to the invalidator.  Public key is passed into
    the validator on startup using the --rpc-adversary-indentity argument.

    If validator was started with --no-rpc-adversary-auth, then this argument
    should be ommited.

    Default: <empty>

  --attack-target {<pubkey> | <IP>}  Used by repair packet flow tests,
    specifying another validator in the network that will be attacked.

    Default: <empty>

  --invalidator-client <solana-invalidator-client path>  Path to a binary that
    is the invalidator client.  Either absolute or from PATH.  Executed as
    specified.

    Default: solana-invalidator-client
EOM
}

#
# === Argument processing ===
#

die() {
  printf 'ERROR: %s' "$1" >&2
  shift
  [[ $# -gt 0 ]] && printf ' %s' "$@" >&2
  printf '\n' >&2

  exit 1
}

requires_arg() {
  local count=$1
  local value=$2
  local name=$3
  local cliName=$4
  local type=$5
  shift 5
  [[ $# -ne 0 ]] && die "required_arg takes 5 arguments"

  if [[ "$count" -le 0 ]]; then
    die "--$name requires an argument: $type"
  fi

  declare -g "$name"
  printf -v "$name" %s "$value"
}

requires_re_arg() {
  local count=$1
  local value=$2
  local name=$3
  local cliName=$4
  local type=$5
  local expectedRe=$6
  shift 6
  [[ $# -ne 0 ]] && die "required_re_arg takes 6 arguments"

  if [[ "$count" -le 0 ]]; then
    die "--$cliName requires an argument: $type"
  fi

  if [[ ! "$value" =~ $expectedRe ]]; then
    die "--$cliName requires an argument: $type. Got: \"$value\""
  fi

  declare -g "$name"
  printf -v "$name" %s "$value"
}

if ! command -v "$invalidatorClient" &>/dev/null; then
  cat <<EOM

Unable to find solana invalidator client.
Specified path: $invalidatorClient
Check the script header at: $me
EOM
  exit 1
fi

error() {
  local error=$1

  if [[ -n "$error" ]]; then
    echo "Error: $error"
    echo
  fi

  help_msg

  if [[ -n "$error" ]]; then
    exit 1
  else
    exit 0
  fi
}

while [[ $# -gt 0 ]]; do
  name=$1
  shift

  case $name in
    --runtime)
      requires_re_arg $# "$1" runtime runtime seconds '^[0-9]+$'
      shift
      ;;
    --sleeptime)
      requires_re_arg $# "$1" sleeptime sleeptime seconds '^[0-9]+$'
      shift
      ;;
    --iterations)
      requires_re_arg $# "$1" iterations iterations 'count of "unbounded"' \
        '^([0-9]+|unbounded)$'
      shift
      ;;
    --rpc-adversary-keypair)
      requires_arg $# "$1" \
        rpcAdversaryKeypair "rpc-adversary-keypair" "file path"
      [[ ! -r "$rpcAdversaryKeypair" ]] && \
        die "--rpc-adversary-keypair must point to an existing file." \
          "Got: $rpcAdversaryKeypair"
      shift
      ;;
    --attack-target)
      requires_arg $# "$1" attackTarget "attack-target" "attack name"
      shift
      ;;
    --invalidator-client)
      requires_arg $# "$1" invalidatorClient "invalidator-client" \
        "path to binary"
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

declare -a commonArgs
declare -a repairShArgs

if [[ -z "$runtime" || "$runtime" -eq 0 ]]; then
  die "--runtime argument is required and must be above zero"
fi

if [[ -z "$sleeptime" || "$sleeptime" -eq 0 ]]; then
  die "--sleeptime argument is required and must be above zero"
fi

if [[ -n "$rpcAdversaryKeypair" ]]; then
  commonArgs+=("--rpc-adversary-keypair" "$rpcAdversaryKeypair")
  repairShArgs+=("--rpc-adversary-keypair" "$rpcAdversaryKeypair")
fi

if [[ -n "$attackTarget" ]]; then
  repairShArgs+=("--attack-target" "$attackTarget")
fi

#
# === Execution ===
#
#
# == Attacks categories and individual attacks ==
#

attack_invalidateLeaderBlock() {
  local kind=$1
  shift
  [[ $# -ne 0 ]] && die "attack_invalidateLeaderBlock takes 1 argument"

  "$invalidatorClient" "${commonArgs[@]}" \
    configure-invalidate-leader-block --invalidation-kind "$kind"
  sleep "$runtime"
  "$invalidatorClient" "${commonArgs[@]}" \
    configure-invalidate-leader-block
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

#
# == main ==
#

run_attacks_all() {
  attack_invalidateLeaderBlock invalidFeePayer
  attack_invalidateLeaderBlock invalidSignature

  attack_dropTurbineVotes

  attack_repairTests minimal_packets
  attack_repairTests ping_cache_overflow
  attack_repairTests unavailable_slots
  attack_repairTests ping_overflow_with_orphan
  # attack_repairTests fake_future_leader_slots \
  #  --iteration-delay-us 10000000 \
  #  --packets-per-iteration 20

  attack_gossipPacketFlood pingCacheOverflow \
    --iteration-delay-us 1000000 \
    --packets-per-peer-per-iteration 10000

  attack_replayStage transferRandom
  attack_replayStage createNonceAccounts
  attack_replayStage allocateRandomLarge
  attack_replayStage allocateRandomSmall
  attack_replayStage chainTransactions

  attack_delayBroadcast

  attack_sendDuplicateBlocks
}

main() {
  # If RUST_LOG is unset, default to info.
  export RUST_LOG=${RUST_LOG:-solana=info,solana_runtime::message_processor=debug}
  # Useful for debugging client crashes.
  # TODO Do we still want this set?
  export RUST_BACKTRACE=1

  # Reduce ancestor hash sample size for smaller cluster size
  "$invalidatorClient" "${commonArgs[@]}" \
    configure-repair-parameters --ancestor-hash-repair-sample-size 2

  local iter=0
  while true; do
    if [[ "$iterations" != unbounded && "$iter" -ge "$iterations" ]]; then
      break
    fi
    iter=$(( iter + 1 ))

    printf 'Iteration: %s\n' "$iter"
    run_attacks_all
  done
}

main
