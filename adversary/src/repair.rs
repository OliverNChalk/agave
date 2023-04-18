use {
    crate::adversary_feature_set::repair_minimal_packet_flood,
    log::*,
    solana_gossip::{cluster_info::ClusterInfo, contact_info::Protocol},
    solana_streamer::sendmmsg::{batch_send, SendPktsError},
    std::{
        net::UdpSocket,
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        },
        thread::{self, sleep, Builder, JoinHandle},
        time::{Duration, Instant},
    },
};

pub struct RepairMinimalPacketFlood {
    exit: Arc<AtomicBool>,
    thread_hdl: JoinHandle<()>,
}

impl RepairMinimalPacketFlood {
    pub fn new(repair_socket: Arc<UdpSocket>, cluster_info: Arc<ClusterInfo>) -> Self {
        let exit = Arc::new(AtomicBool::default());
        let thread_hdl = {
            let exit = exit.clone();
            Builder::new()
                .name("solAdvRepMinFld".to_string())
                .spawn(move || {
                    Self::run(&repair_socket, &cluster_info, exit);
                })
                .unwrap()
        };
        Self { exit, thread_hdl }
    }

    pub fn join(self) -> thread::Result<()> {
        self.exit.store(true, Ordering::Relaxed);
        self.thread_hdl.join()
    }

    fn run(repair_socket: &UdpSocket, cluster_info: &ClusterInfo, exit: Arc<AtomicBool>) {
        const REPORT_INTERVAL_MS: u64 = 2_000;
        let mut last_report = Instant::now();
        let mut iteration: u64 = 0;
        let mut packet_count: usize = 0;

        loop {
            if exit.load(Ordering::Relaxed) {
                return;
            }
            let config = repair_minimal_packet_flood::get_config();
            let peers = cluster_info.repair_peers(u64::MAX);
            let mut reqs_v = vec![];
            peers.iter().for_each(|peer| {
                for i in 0..config.packets_per_peer_per_iteration {
                    let val: u8 = i.wrapping_rem(u8::MAX as u32).try_into().unwrap();
                    reqs_v.push((vec![val], peer.serve_repair(Protocol::UDP).unwrap()));
                }
            });
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
            if last_report.elapsed() >= Duration::from_millis(REPORT_INTERVAL_MS) {
                info!(
                    "repair_minimal_packet_flood num_peers={} itreation={iteration}, \
                     packets={packet_count}",
                    peers.len()
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
