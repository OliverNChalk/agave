# shellcheck disable=SC2148
#
# Private invalidator testnet configuration and test plan. This script is
# expected to be conditionally sourced by the continuous-test.sh script.
#

set -o errexit
set -o nounset
set -o pipefail

for requiredVar in commonArgs invalidatorClient; do
  if ! declare -p "$requiredVar" >/dev/null; then
    cat <<EOM
private-testnet expects $requiredVar to be set.  Functions defined in this
file are defined with an assumption that this variable is a global variable
defined by the script that includes this script via the 'source' command.
EOM
    exit 1
  fi
done

run_config() {
  # Reduce ancestor hash sample size for smaller cluster size
  # shellcheck disable=SC2154
  "$invalidatorClient" "${commonArgs[@]}" \
    configure-repair-parameters --ancestor-hash-repair-sample-size 2
}

run_attacks_all() {
  attack_invalidateLeaderBlock invalidFeePayer
  attack_invalidateLeaderBlock invalidSignature

  attack_dropTurbineVotes

  attack_repairTests minimal_packets
  attack_repairTests ping_cache_overflow
  attack_repairTests unavailable_slots
  attack_repairTests ping_overflow_with_orphan
  attack_repairTests fake_future_leader_slots \
   --iteration-delay-us 10000000 \
   --packets-per-iteration 20

  attack_gossipPacketFlood pingCacheOverflow \
    --iteration-delay-us 1000000 \
    --packets-per-peer-per-iteration 10000

  # attacks that use only fee payer accounts
  attack_replayStage transferRandom
  attack_replayStage transferRandomWithMemo
  attack_replayStage createNonceAccounts
  attack_replayStage allocateRandomLarge
  attack_replayStage allocateRandomSmall
  attack_replayStage chainTransactions
  # attacks that use max accounts
  attack_replayStage readMaxAccounts
  attack_replayStage writeMaxAccounts
  # attacks that execute deployed program
  # how many transactions can be processes in parallel by one replay thread.
  local BATCH_SIZE=64
  # One tx can load up to 64MB, each account is 10MB.
  # Hence, 6 is max value we can use.
  # Temporarily reducing value to 1 because slot replay times currently explode
  # to minutes with 6.
  local NUM_ACCOUNTS_PER_TX=1
  # Transaction budget to prevent transaction from being successful.
  # Required because `use-failed-transaction-hotpath` txs must not succeed.
  local INSUFFICIENT_CU_BUDGET=100
  attack_replayStage largeNop \
   --transaction-batch-size ${BATCH_SIZE}
  # Temporarily reducing value to 1 because slot replay times currently explode
  # to minutes with 64.
  BATCH_SIZE=1
  attack_replayStage writeProgram \
   --transaction-batch-size ${BATCH_SIZE} \
   --num-accounts-per-tx ${NUM_ACCOUNTS_PER_TX} \
   --transaction-cu-budget ${INSUFFICIENT_CU_BUDGET} \
   --use-failed-transaction-hotpath
  attack_replayStage readProgram \
   --transaction-batch-size ${BATCH_SIZE} \
   --num-accounts-per-tx ${NUM_ACCOUNTS_PER_TX} \
   --transaction-cu-budget ${INSUFFICIENT_CU_BUDGET} \
   --use-failed-transaction-hotpath
  # for txs to succeed we need at least 176k CU
  attack_replayStage recursiveProgram \
   --transaction-batch-size ${BATCH_SIZE} \
   --num-accounts-per-tx ${NUM_ACCOUNTS_PER_TX} \
   --transaction-cu-budget ${INSUFFICIENT_CU_BUDGET} \
   --use-failed-transaction-hotpath
  # attack that executes numerous deployed programs
  # which triggers recompilations
  attack_replayStage coldProgramCache \
   --transaction-batch-size ${BATCH_SIZE} \
   --num-accounts-per-tx ${NUM_ACCOUNTS_PER_TX} \
   --transaction-cu-budget ${INSUFFICIENT_CU_BUDGET} \
   --use-failed-transaction-hotpath

  attack_delayBroadcast

  attack_sendDuplicateBlocks
  # Temporarily disabled because this test will fail consistently until we have
  # retransmit signatures
  # attack_sendDuplicateLeafNodes

  attack_unloadProgramInvocation
}
