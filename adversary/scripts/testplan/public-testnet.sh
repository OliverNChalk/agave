# shellcheck disable=SC2148
#
# Public testnet configuration and test plan. This script is expected to be
# conditionally sourced by the continuous-test.sh script.
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

  # Temporarily disabled because this test is capable of bringing down several nodes
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
