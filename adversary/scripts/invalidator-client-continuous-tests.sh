#!/usr/bin/env bash
#
# Loop through invalidator test cases with solana-invalidator-client.
# Ctrl-C to exit.
#

BIN=solana-invalidator-client
SCRIPT_DIR="$(readlink -f "$(dirname "$0")")"

which $BIN > /dev/null 2>&1 || {
  echo
  echo "Unable to locate $BIN."
  echo
  exit 1
}

export RUST_LOG=${RUST_LOG:-solana=info,solana_runtime::message_processor=debug} # if RUST_LOG is unset, default to info
export RUST_BACKTRACE=1

RUNTIME=$1
SLEEPTIME=$2
ITERATIONS=$3

if [[ -z $RUNTIME || -z $SLEEPTIME ]]; then
  echo "Usage: $0 <RUNTIME secs> <SLEEPTIME secs> [<ITERATIONS>]"
  exit 1
fi

# Reduce ancestor hash sample size for smaller cluster size
$BIN configure-repair-parameters --ancestor-hash-repair-sample-size 2

commands=(
  "$BIN configure-invalidate-leader-block \
    --invalidation-kind invalidFeePayer"
  "sleep $RUNTIME"
  "$BIN configure-invalidate-leader-block"
  "sleep $SLEEPTIME"
  "$BIN configure-invalidate-leader-block \
    --invalidation-kind invalidSignature"
  "sleep $RUNTIME"
  "$BIN configure-invalidate-leader-block"
  "sleep $SLEEPTIME"
  "$BIN configure-drop-turbine-votes \
    --drop true"
  "sleep $RUNTIME"
  "$BIN configure-drop-turbine-votes \
    --drop false"
  "sleep $SLEEPTIME"
  "$SCRIPT_DIR/repair-tests.sh minimal_packets"
  "sleep $RUNTIME"
  "$SCRIPT_DIR/repair-tests.sh disable"
  "sleep $SLEEPTIME"
  "$SCRIPT_DIR/repair-tests.sh ping_cache_overflow"
  "sleep $RUNTIME"
  "$SCRIPT_DIR/repair-tests.sh disable"
  "sleep $SLEEPTIME"
  "$SCRIPT_DIR/repair-tests.sh unavailable_slots"
  "sleep $RUNTIME"
  "$SCRIPT_DIR/repair-tests.sh disable"
  "sleep $SLEEPTIME"
  "$SCRIPT_DIR/repair-tests.sh ping_overflow_with_orphan"
  "sleep $RUNTIME"
  "$SCRIPT_DIR/repair-tests.sh disable"
  "sleep $SLEEPTIME"
  "$BIN configure-gossip-packet-flood \
    --flood-strategy pingCacheOverflow \
    --iteration-delay-us 1000000 \
    --packets-per-peer-per-iteration 10000"
  "sleep $RUNTIME"
  "$BIN configure-gossip-packet-flood"
  "sleep $SLEEPTIME"
  "$BIN configure-replay-stage-attack \
    --selected-attack transferRandom"
  "sleep $RUNTIME"
  "$BIN configure-replay-stage-attack"
  "sleep $SLEEPTIME"
  "$BIN configure-replay-stage-attack \
    --selected-attack createNonceAccounts"
  "sleep $RUNTIME"
  "$BIN configure-replay-stage-attack"
  "sleep $SLEEPTIME"
  "$BIN configure-replay-stage-attack \
    --selected-attack allocateRandomLarge"
  "sleep $RUNTIME"
  "$BIN configure-replay-stage-attack"
  "sleep $SLEEPTIME"
  "$BIN configure-replay-stage-attack \
    --selected-attack allocateRandomSmall"
  "sleep $RUNTIME"
  "$BIN configure-replay-stage-attack"
  "sleep $SLEEPTIME"
  "$BIN configure-replay-stage-attack \
    --selected-attack chainTransactions"
  "sleep $RUNTIME"
  "$BIN configure-replay-stage-attack"
  "sleep $SLEEPTIME"
)
num_commands=${#commands[@]}

if [[ -z $ITERATIONS ]]; then
  ITERATIONS=0
else
  ITERATIONS=$((ITERATIONS * num_commands))
fi

i=0
while [ $ITERATIONS -eq 0 ] || [ $i -lt $ITERATIONS ]; do
  echo "Iteration $i"
  ${commands[$((i % num_commands))]}
  ((i++))
done
