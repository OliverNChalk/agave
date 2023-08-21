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

TESTCASE=$1

usage() {
    echo "Usage: $0 [minimal_packets | signed_packets | ping_cache_overflow"
    echo "  | orphan | fake_future_leader_slots | disable]"
}

if [[ -z $TESTCASE ]]; then
    usage
    exit 1
fi

case $TESTCASE in
minimal_packets)
  $BIN configure-repair-packet-flood \
    --flood-strategy minimalPackets \
    --iteration-delay-us 100000 \
    --packets-per-peer-per-iteration 10000
  ;;

signed_packets)
  $BIN configure-repair-packet-flood \
    --flood-strategy signedPackets \
    --iteration-delay-us 100000 \
    --packets-per-peer-per-iteration 10000
  ;;

ping_cache_overflow)
  $BIN configure-repair-packet-flood \
    --flood-strategy pingCacheOverflow \
    --iteration-delay-us 10000 \
    --packets-per-peer-per-iteration 10000
  ;;

orphan)
  $BIN configure-repair-packet-flood \
    --flood-strategy orphan \
    --iteration-delay-us 100000 \
    --packets-per-peer-per-iteration 10000
  ;;

fake_future_leader_slots)
  $BIN configure-repair-packet-flood \
    --flood-strategy fakeFutureLeaderSlots \
    --iteration-delay-us 1000000 \
    --packets-per-peer-per-iteration 1000
  ;;

unavailable_slots)
  $BIN configure-repair-packet-flood \
    --flood-strategy unavailableSlots \
    --iteration-delay-us 100000 \
    --packets-per-peer-per-iteration 10000
  ;;

ping_overflow_with_drop)
  $BIN configure-packet-drop-parameters \
    --broadcast-packet-drop-percent 20 \
    --retransmit-packet-drop-percent 20
  $BIN configure-repair-packet-flood \
    --flood-strategy pingCacheOverflow \
    --iteration-delay-us 10000 \
    --packets-per-peer-per-iteration 10000
  ;;

ping_overflow_with_orphan)
  $BIN configure-repair-packet-flood --toml - <<EOF
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
  $BIN configure-repair-packet-flood
  $BIN configure-packet-drop-parameters
  ;;

*)
  echo "Invalid test case."
  echo
  usage
  exit 1
  ;;
esac
