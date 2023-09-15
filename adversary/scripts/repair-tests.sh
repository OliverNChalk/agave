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
$0 --test [minimal_packets | signed_packets | ping_cache_overflow |
  orphan | fake_future_leader_slots | disable]
  [--rpc-adversary-keypair <keypair path>]
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

case $TESTCASE in
minimal_packets)
  $BIN "$COMMON_ARGS" configure-repair-packet-flood \
    --flood-strategy minimalPackets \
    --iteration-delay-us 100000 \
    --packets-per-peer-per-iteration 10000
  ;;

signed_packets)
  $BIN "$COMMON_ARGS" configure-repair-packet-flood \
    --flood-strategy signedPackets \
    --iteration-delay-us 100000 \
    --packets-per-peer-per-iteration 10000
  ;;

ping_cache_overflow)
  $BIN "$COMMON_ARGS" configure-repair-packet-flood \
    --flood-strategy pingCacheOverflow \
    --iteration-delay-us 10000 \
    --packets-per-peer-per-iteration 10000
  ;;

orphan)
  $BIN "$COMMON_ARGS" configure-repair-packet-flood \
    --flood-strategy orphan \
    --iteration-delay-us 100000 \
    --packets-per-peer-per-iteration 10000
  ;;

fake_future_leader_slots)
  $BIN "$COMMON_ARGS" configure-repair-packet-flood \
    --flood-strategy fakeFutureLeaderSlots \
    --iteration-delay-us 1000000 \
    --packets-per-peer-per-iteration 1000
  ;;

unavailable_slots)
  $BIN "$COMMON_ARGS" configure-repair-packet-flood \
    --flood-strategy unavailableSlots \
    --iteration-delay-us 100000 \
    --packets-per-peer-per-iteration 10000
  ;;

ping_overflow_with_drop)
  $BIN "$COMMON_ARGS" configure-packet-drop-parameters \
    --broadcast-packet-drop-percent 20 \
    --retransmit-packet-drop-percent 20
  $BIN "$COMMON_ARGS" configure-repair-packet-flood \
    --flood-strategy pingCacheOverflow \
    --iteration-delay-us 10000 \
    --packets-per-peer-per-iteration 10000
  ;;

ping_overflow_with_orphan)
  $BIN "$COMMON_ARGS" configure-repair-packet-flood --toml - <<EOF
[[configs]]
floodStrategy = "pingCacheOverflow"
packetsPerPeerPerIteration = 10000
iterationDelayUs = 10000
[[configs]]
floodStrategy = "orphan"
packetsPerPeerPerIteration = 10000
iterationDelayUs = 10000
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
