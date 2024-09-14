# shellcheck disable=SC2148
#
# Private invalidator testnet configuration and test plan that does not require
# any accounts or programs to be setup in advance. This script is expected to be
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

  attack_delayBroadcast

  attack_sendDuplicateBlocks
  attack_sendDuplicateLeafNodes
}
