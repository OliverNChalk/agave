#!/usr/bin/env bash
#
# Loop through invalidator test cases with solana-invalidator-client.
# Ctrl-C to exit.
#

BIN=solana-invalidator-client

which $BIN || {
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

commands=(
  "$BIN configure-send-duplicate-blocks \
    --new-entry-index-from-end 0 \
    --num-duplicate-validators 1 \
    --send-original-after-ms 0"
  "sleep $RUNTIME"
  "$BIN configure-send-duplicate-blocks"
  "sleep $SLEEPTIME"
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
  "$BIN configure-repair-packet-flood \
    --flood-strategy minimalPackets \
    --iteration-delay-us 1000000 \
    --packets-per-peer-per-iteration 10"
  "sleep $RUNTIME"
  "$BIN configure-repair-packet-flood \
    --disable"
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
