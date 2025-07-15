# Implemented attacks

[Replay attacks](./replay-attacks.md)

## Flood attacks

Spamming attacks targetting validator ports

### Shred sigverify flood

Attack flood validator TVU port with invalid shreds in an attempt to increase memory load
and potentially cause an OOM.

The most effective strategy is to create dummy shreds that are structured in a way that they
get to the shred_sigverify stage. This requires bypassing the dedupper and the checks in the
shred_fetch stage.

Most of the checks in the shred_fetch stage are in should_discard_shred():
1. shred slot is between root and max_slot+1
    - max_slot = last_slot + MAX_SHRED_DISTANCE_MINIMUM.max(2 * slots_per_epoch);
2. must be well-formed enough so shred, version, variant, slot, index, etc can be parsed out
3. version must equal the current shred_version
4. for data shreds: index must be < MAX_DATA_SHREDS_PER_SLOT
5. for code shreds: index must be < MAX_CODE_SHREDS_PER_SLOT
6. slot value must be higher than parent slot value
7. parent must be >= root

These are all relatively easy to satisfy.

There are a lot of potential ways to bypass the dedupper but the easiest I found was to send
in the same shred but sample the slot index from a random distribution, within the ranges defined
above.

With enough load, this will exhaust the target validator's sigverify capacity, requiring the
input channel to either block or load-shed if bounded, or grow without bound otherwise.

### Gossip sigverify flood

Attack floods validator gossip port with gossip messages containing valid CrdsValues in an attempt
to overwhelm the gossip signature verification pipeline and potentially cause resource exhaustion.

The attack creates properly structured gossip Protocol::PullResponse messages containing multiple
CrdsValues of type EpochSlots. Each CrdsValue is cryptographically signed with a unique keypair,
making them pass initial structural validation but consuming significant CPU resources during
signature verification.

Key characteristics:
1. Targets the validator's gossip port (typically 8000/UDP)
2. Uses Protocol::PullResponse messages which are processed without immediate filtering
3. Each packet contains 10 valid CrdsValues to maximize verification overhead per packet
4. CrdsValues use EpochSlots data type because they all packing more signatures
5. Uses unique keypairs per CrdsValue to avoid simple deduplication

With sufficient load, this exhausts the target validator's gossip signature verification capacity,
potentially causing:
- Gossip message processing delays
- Resource exhaustion in the gossip service threads, which may bleed into other validator components
- Degraded network synchronization with legitimate peers