# shellcheck disable=SC2148
#
# Private invalidator testnet configuration and test plan. This script is
# expected to be conditionally sourced by the continuous-test.sh script.
#

set -o errexit
set -o nounset
set -o pipefail

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

  attack_gossipPacketFlood pullContactInfo \
    --iteration-delay-us 1000000 \
    --packets-per-peer-per-iteration 10000

  attack_gossipPacketFlood pushContactInfo \
    --iteration-delay-us 1000000 \
    --packets-per-peer-per-iteration 10000

  attack_tpuPacketFlood udpVoteOverflow \
    --iteration-duration-us 0

  # attacks that use only fee payer accounts
  attack_replayStage transferRandom
  attack_replayStage transferRandomWithMemo
  attack_replayStage createNonceAccounts
  attack_replayStage allocateRandomLarge
  attack_replayStage allocateRandomSmall
  attack_replayStage chainTransactions
  attack_replayStage readNonExistentAccounts
  # attacks that use max accounts
  attack_replayStage readMaxAccounts
  attack_replayStage writeMaxAccounts
  # attacks that execute deployed program
  attack_replayStage largeNop
  # One tx can load up to 64MB, each account is 10MB.
  # Hence, 6 is max value we can use.
  local NUM_ACCOUNTS_PER_TX=6
  # Transaction budget to prevent transaction from being successful.
  # Required because `use-failed-transaction-hotpath` txs must not succeed.
  local INSUFFICIENT_CU_BUDGET=100
  local BATCH_SIZE=64
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
  # Increase the CU budget to allow further progress into the program before
  # failing. Currently each tx consumes a little over 1M CUs.
  local INSUFFICIENT_CU_BUDGET=1000000
  attack_replayStage cpiProgram \
   --transaction-batch-size ${BATCH_SIZE} \
   --num-accounts-per-tx ${NUM_ACCOUNTS_PER_TX} \
   --transaction-cu-budget ${INSUFFICIENT_CU_BUDGET} \
   --use-failed-transaction-hotpath

  attack_delayBroadcast

  attack_sendDuplicateBlocks
  attack_sendDuplicateLeafNodes

  attack_unloadProgramInvocation

  # attacks that use invalid fee payers.
  attack_replayStage readNonExistentAccounts \
   --use-failed-transaction-hotpath \
   --use-invalid-fee-payer
}
