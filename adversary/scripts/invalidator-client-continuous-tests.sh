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

help_msg () {
  cat <<EOM
$0 --runtime <runtime secs> --sleeptime <sleeptime secs>
  [--iterations <num iterations>] [--rpc-adversary-keypair <keypair path>]
  [--attack-target {pubkey | IP}]
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
    --runtime)
      RUNTIME="$2"
      shift 2
      ;;
    --sleeptime)
      SLEEPTIME="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --rpc-adversary-keypair)
      KEYPAIR="$2"
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

if [ -z "$RUNTIME" ]; then
  help "--runtime argument is required"
fi

if [ -z "$SLEEPTIME" ]; then
  help "--sleeptime argument is required"
fi

if [ -n "$KEYPAIR" ]; then
  COMMON_ARGS="--rpc-adversary-keypair $KEYPAIR"
  REPAIR_SH_ARGS="$COMMON_ARGS"
fi

if [ -n "$ATTACK_TARGET" ]; then
  REPAIR_SH_ARGS="$REPAIR_SH_ARGS --attack-target $ATTACK_TARGET"
fi

# Reduce ancestor hash sample size for smaller cluster size
$BIN "$COMMON_ARGS" configure-repair-parameters --ancestor-hash-repair-sample-size 2

commands=(
  "$BIN $COMMON_ARGS configure-invalidate-leader-block \
    --invalidation-kind invalidFeePayer"
  "sleep $RUNTIME"
  "$BIN $COMMON_ARGS configure-invalidate-leader-block"
  "sleep $SLEEPTIME"
  "$BIN $COMMON_ARGS configure-invalidate-leader-block \
    --invalidation-kind invalidSignature"
  "sleep $RUNTIME"
  "$BIN $COMMON_ARGS configure-invalidate-leader-block"
  "sleep $SLEEPTIME"
  "$BIN $COMMON_ARGS configure-drop-turbine-votes \
    --drop true"
  "sleep $RUNTIME"
  "$BIN $COMMON_ARGS configure-drop-turbine-votes \
    --drop false"
  "sleep $SLEEPTIME"
  "$SCRIPT_DIR/repair-tests.sh $REPAIR_SH_ARGS --test minimal_packets"
  "sleep $RUNTIME"
  "$SCRIPT_DIR/repair-tests.sh $REPAIR_SH_ARGS --test disable"
  "sleep $SLEEPTIME"
  "$SCRIPT_DIR/repair-tests.sh $REPAIR_SH_ARGS --test ping_cache_overflow"
  "sleep $RUNTIME"
  "$SCRIPT_DIR/repair-tests.sh $REPAIR_SH_ARGS --test disable"
  "sleep $SLEEPTIME"
  "$SCRIPT_DIR/repair-tests.sh $REPAIR_SH_ARGS --test unavailable_slots"
  "sleep $RUNTIME"
  "$SCRIPT_DIR/repair-tests.sh $REPAIR_SH_ARGS --test disable"
  "sleep $SLEEPTIME"
  "$SCRIPT_DIR/repair-tests.sh $REPAIR_SH_ARGS --test ping_overflow_with_orphan"
  "sleep $RUNTIME"
  "$SCRIPT_DIR/repair-tests.sh $REPAIR_SH_ARGS --test disable"
  "sleep $SLEEPTIME"
  "$BIN $COMMON_ARGS configure-gossip-packet-flood \
    --flood-strategy pingCacheOverflow \
    --iteration-delay-us 1000000 \
    --packets-per-peer-per-iteration 10000"
  "sleep $RUNTIME"
  "$BIN $COMMON_ARGS configure-gossip-packet-flood"
  "sleep $SLEEPTIME"
  "$BIN $COMMON_ARGS configure-replay-stage-attack \
    --selected-attack transferRandom"
  "sleep $RUNTIME"
  "$BIN $COMMON_ARGS configure-replay-stage-attack"
  "sleep $SLEEPTIME"
  "$BIN $COMMON_ARGS configure-replay-stage-attack \
    --selected-attack createNonceAccounts"
  "sleep $RUNTIME"
  "$BIN $COMMON_ARGS configure-replay-stage-attack"
  "sleep $SLEEPTIME"
  "$BIN $COMMON_ARGS configure-replay-stage-attack \
    --selected-attack allocateRandomLarge"
  "sleep $RUNTIME"
  "$BIN $COMMON_ARGS configure-replay-stage-attack"
  "sleep $SLEEPTIME"
  "$BIN $COMMON_ARGS configure-replay-stage-attack \
    --selected-attack allocateRandomSmall"
  "sleep $RUNTIME"
  "$BIN $COMMON_ARGS configure-replay-stage-attack"
  "sleep $SLEEPTIME"
  "$BIN $COMMON_ARGS configure-replay-stage-attack \
    --selected-attack chainTransactions"
  "sleep $RUNTIME"
  "$BIN $COMMON_ARGS configure-replay-stage-attack"
  "sleep $SLEEPTIME"
  "$BIN $COMMON_ARGS configure-send-duplicate-blocks \
    --turbine-send-delay-ms 1600"
  "sleep $RUNTIME"
  "$BIN $COMMON_ARGS configure-send-duplicate-blocks"
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
