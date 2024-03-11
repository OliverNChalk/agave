# Inimica

Inimica performs attacks on the network as an external client.

The implementation is in the [`invalidator/inimica`](../../invalidator/inimica/README.md) folder.

Currently the following attacks are defined.

## Program runtime attacks

### Program cache attacks

#### Invocation of an unloaded program

For program loader v2 Solana cache used to have a bug, that caused the validators to enter infinite
loop under certain circumstances.

More specifically, it was enough to do the following:

1. Deploy a program with the v2 program loader (`BPFLoader2111111111111111111111111111111111`).

   This is supposed to create an entry in the validator cached, holding the program ELF, and the
   slot number where the program was deployed.

2. Wait for the program to be removed from the cache.  There is no command to do this outside of a
   specific validator, but there is a good chance a program is removed from the cache if other
   programs are deployed and/or invoked.

   At this point, an entry for the program in the cache is expected to be replaced with an
   "unloaded" entry, recording the deployment slot.

3. Call the deployed program.  When it is invoked, the program is loaded again, but for v1 and v2
   loaders, the deployment slot is not recorded in the program data and is set, conservatively, to
   0.

   The bug is that an existing unloaded entry then has a newer slot number, and the new entry
   created in this step is not added to the cache.  The validator then asks the cache for the
   program entry, finds that it is unloaded and tries again.

   Loading happens in a loop with no time or count restrictions and the above logic causes the
   validator to enter an infinite loop.

For v3 (`BPFLoaderUpgradeab1e11111111111111111111111`) and
v4(`LoaderV411111111111111111111111111111111111`) loaders, the problem does not occur, as the
deployment contains the deployment slot, and the loaded entry created from it then contains the same
slot as the unloaded entry.  Resulting in the loaded entry being correctly added to the cache.

It still does not hurt to exercise this scenario with later loaders, as the caching logic is still
somewhat non-trivial.
