#!/usr/bin/env bash
#
# Invalidator repair related test cases
#

BIN=solana-invalidator-client

which $BIN > /dev/null 2>&1 || {
  echo
  echo "Unable to locate $BIN."
  echo
  exit 1
}

export RUST_LOG=${RUST_LOG:-solana=info,solana_runtime::message_processor=debug} # if RUST_LOG is unset, default to info
export RUST_BACKTRACE=1

help_msg () {
  cat <<EOM
$0 --test {minimal_packets | signed_packets | ping_cache_overflow
  | orphan | fake_future_leader_slots | unavailable_slots
  | ping_overflow_with_drop | ping_overflow_with_orphan | disable}
  [--rpc-adversary-keypair <keypair path>]
  [--attack-target {pubkey | IP}]
  [--iteration-delay-us <delay in us>]
  [--packets-per-iteration <num packets>]
EOM
}

help () {
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
  case $1 in
    --test)
      if [ -n "$TESTCASE" ]; then
        help "Error: --test $TESTCASE already defined"
      fi
      TESTCASE="$2"
      shift 2
      ;;
    --rpc-adversary-keypair)
      KEYPAIR="$2"
      shift 2
      ;;
    --iteration-delay-us)
      ITERATION_DELAY="$2"
      shift 2
      ;;
    --packets-per-iteration)
      PACKETS_PER_ITERATION="$2"
      shift 2
      ;;
    --attack-target)
      ATTACK_TARGET="$2"
      shift 2
      ;;
    --help)
      help
      ;;
    *)
      help "Unknown argument $1"
      ;;
  esac
done

if [ -z "$TESTCASE" ]; then
  help "Error: --test argument is required"
fi

if [ -n "$KEYPAIR" ]; then
  COMMON_ARGS="--rpc-adversary-keypair $KEYPAIR"
fi

if [ -n "$ATTACK_TARGET" ]; then
  ATTACK_TARGET_PARAM="--target $ATTACK_TARGET"
  ATTACK_TARGET_TOML_LINE="target = $ATTACK_TARGET"
fi

case $TESTCASE in
minimal_packets)
  [ -z "$ITERATION_DELAY" ] && ITERATION_DELAY="100000"
  [ -z "$PACKETS_PER_ITERATION" ] && PACKETS_PER_ITERATION="10000"
  $BIN "$COMMON_ARGS" configure-repair-packet-flood \
    --flood-strategy minimalPackets \
    --iteration-delay-us "$ITERATION_DELAY" \
    --packets-per-peer-per-iteration "$PACKETS_PER_ITERATION" \
    "$ATTACK_TARGET_PARAM"
  ;;

signed_packets)
  [ -z "$ITERATION_DELAY" ] && ITERATION_DELAY="100000"
  [ -z "$PACKETS_PER_ITERATION" ] && PACKETS_PER_ITERATION="10000"
  $BIN "$COMMON_ARGS" configure-repair-packet-flood \
    --flood-strategy signedPackets \
    --iteration-delay-us "$ITERATION_DELAY" \
    --packets-per-peer-per-iteration "$PACKETS_PER_ITERATION" \
    "$ATTACK_TARGET_PARAM"
  ;;

ping_cache_overflow)
  [ -z "$ITERATION_DELAY" ] && ITERATION_DELAY="10000"
  [ -z "$PACKETS_PER_ITERATION" ] && PACKETS_PER_ITERATION="10000"
  $BIN "$COMMON_ARGS" configure-repair-packet-flood \
    --flood-strategy pingCacheOverflow \
    --iteration-delay-us "$ITERATION_DELAY" \
    --packets-per-peer-per-iteration "$PACKETS_PER_ITERATION" \
    "$ATTACK_TARGET_PARAM"
  ;;

orphan)
  [ -z "$ITERATION_DELAY" ] && ITERATION_DELAY="100000"
  [ -z "$PACKETS_PER_ITERATION" ] && PACKETS_PER_ITERATION="10000"
  $BIN "$COMMON_ARGS" configure-repair-packet-flood \
    --flood-strategy orphan \
    --iteration-delay-us "$ITERATION_DELAY" \
    --packets-per-peer-per-iteration "$PACKETS_PER_ITERATION" \
    "$ATTACK_TARGET_PARAM"
  ;;

fake_future_leader_slots)
  [ -z "$ITERATION_DELAY" ] && ITERATION_DELAY="1000000"
  [ -z "$PACKETS_PER_ITERATION" ] && PACKETS_PER_ITERATION="1000"
  $BIN "$COMMON_ARGS" configure-repair-packet-flood \
    --flood-strategy fakeFutureLeaderSlots \
    --iteration-delay-us "$ITERATION_DELAY" \
    --packets-per-peer-per-iteration "$PACKETS_PER_ITERATION" \
    "$ATTACK_TARGET_PARAM"
  ;;

unavailable_slots)
  [ -z "$ITERATION_DELAY" ] && ITERATION_DELAY="100000"
  [ -z "$PACKETS_PER_ITERATION" ] && PACKETS_PER_ITERATION="10000"
  $BIN "$COMMON_ARGS" configure-repair-packet-flood \
    --flood-strategy unavailableSlots \
    --iteration-delay-us "$ITERATION_DELAY" \
    --packets-per-peer-per-iteration "$PACKETS_PER_ITERATION" \
    "$ATTACK_TARGET_PARAM"
  ;;

ping_overflow_with_drop)
  [ -z "$ITERATION_DELAY" ] && ITERATION_DELAY="10000"
  [ -z "$PACKETS_PER_ITERATION" ] && PACKETS_PER_ITERATION="10000"
  $BIN "$COMMON_ARGS" configure-packet-drop-parameters \
    --broadcast-packet-drop-percent 20 \
    --retransmit-packet-drop-percent 20
  $BIN "$COMMON_ARGS" configure-repair-packet-flood \
    --flood-strategy pingCacheOverflow \
    --iteration-delay-us "$ITERATION_DELAY" \
    --packets-per-peer-per-iteration "$PACKETS_PER_ITERATION" \
    "$ATTACK_TARGET_PARAM"
  ;;

ping_overflow_with_orphan)
  [ -z "$ITERATION_DELAY" ] && ITERATION_DELAY="10000"
  [ -z "$PACKETS_PER_ITERATION" ] && PACKETS_PER_ITERATION="10000"
  $BIN "$COMMON_ARGS" configure-repair-packet-flood --toml - <<EOF
[[configs]]
floodStrategy = "pingCacheOverflow"
packetsPerPeerPerIteration = $PACKETS_PER_ITERATION
iterationDelayUs = $ITERATION_DELAY
${ATTACK_TARGET_TOML_LINE}
[[configs]]
floodStrategy = "orphan"
packetsPerPeerPerIteration = $PACKETS_PER_ITERATION
iterationDelayUs = $ITERATION_DELAY
${ATTACK_TARGET_TOML_LINE}
EOF
  ;;

disable)
  $BIN "$COMMON_ARGS" configure-repair-packet-flood
  $BIN "$COMMON_ARGS" configure-packet-drop-parameters
  ;;

*)
  help "Invalid test case."
  ;;
esac
