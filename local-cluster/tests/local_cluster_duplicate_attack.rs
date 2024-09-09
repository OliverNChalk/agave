use {
    assert_matches::assert_matches,
    reqwest::blocking::Client,
    serde::{Deserialize, Serialize},
    serial_test::serial,
    solana_adversary::adversary_feature_set::send_duplicate_blocks::AdversarialConfig,
    solana_cluster_type::ClusterType,
    solana_commitment_config::CommitmentConfig,
    solana_core::repair::duplicate_repair_status::set_ancestor_hash_repair_sample_size_for_tests_only,
    solana_epoch_schedule::{MAX_LEADER_SCHEDULE_EPOCH_OFFSET, MINIMUM_SLOTS_PER_EPOCH},
    solana_gossip::gossip_service::discover_validators,
    solana_local_cluster::{
        cluster_tests,
        integration_tests::{DEFAULT_NODE_STAKE, RUST_LOG_FILTER},
        local_cluster::{ClusterConfig, LocalCluster, DEFAULT_MINT_LAMPORTS},
    },
    solana_rpc_client::rpc_client::RpcClient,
    solana_streamer::socket::SocketAddrSpace,
    std::{thread::sleep, time::Duration},
};

// Enough nodes to generate a multi-layer turbine tree while not increasing
// execution time too much.
const NUM_NODES: usize = 4;

fn configure_duplicate_block(
    cluster: &LocalCluster,
    config: AdversarialConfig,
) -> Result<(), String> {
    let url = &cluster_tests::get_rpc_url(cluster);
    #[derive(Debug, Serialize, Deserialize)]
    struct RpcRequest {
        jsonrpc: String,
        method: String,
        params: serde_json::Value,
        id: u64,
    }

    let rpc_method = "configureSendDuplicateBlocks";
    let params = serde_json::json!([config]);
    let payload = RpcRequest {
        jsonrpc: "2.0".to_string(),
        method: rpc_method.to_string(),
        params,
        id: 1,
    };
    let client = Client::new();
    let response = client
        .post(url)
        .json(&payload)
        .timeout(Duration::from_secs(10))
        .send()
        .map_err(|e| format!("RPC Send Error: {e:?}"))?;
    response
        .json::<serde_json::Value>()
        .map_err(|e| format!("RPC Parse Response Error: {e:?}"))?;
    Ok(())
}

fn disable_duplicate_block_attack(cluster: &LocalCluster) -> Result<(), String> {
    configure_duplicate_block(cluster, AdversarialConfig::default())
}

fn enable_duplicate_block_attack(cluster: &LocalCluster) -> Result<(), String> {
    let config = AdversarialConfig {
        new_entry_index_from_end: 2,
        leaf_node_partitions: Some(2),
        local_test_pubkey_to_perform_attack: Some(*cluster.entry_point_info.pubkey()),
        ..AdversarialConfig::default()
    };

    configure_duplicate_block(cluster, config)
}

#[test]
#[serial]
fn test_duplicate_block_leaf_node_delivery() {
    solana_logger::setup_with_default(RUST_LOG_FILTER);

    // Setup cluster and connections.
    let mut config =
        ClusterConfig::new_with_equal_stakes(NUM_NODES, DEFAULT_MINT_LAMPORTS, DEFAULT_NODE_STAKE);
    config.cluster_type = ClusterType::Development;
    config.slots_per_epoch = MINIMUM_SLOTS_PER_EPOCH;
    config.stakers_slot_offset =
        MINIMUM_SLOTS_PER_EPOCH.saturating_mul(MAX_LEADER_SCHEDULE_EPOCH_OFFSET);
    let cluster = LocalCluster::new(&mut config, SocketAddrSpace::Unspecified);
    let cluster_nodes = discover_validators(
        &cluster.entry_point_info.gossip().unwrap(),
        NUM_NODES,
        cluster.entry_point_info.shred_version(),
        SocketAddrSpace::Unspecified,
    )
    .unwrap();
    assert_eq!(cluster_nodes.len(), NUM_NODES);

    // Need to force this constant low because of the small number of nodes in
    // this local cluster. Otherwise, we will deadlock when trying to sample
    // hashes to determine which version of duplicate block we want to repair.
    set_ancestor_hash_repair_sample_size_for_tests_only(1);

    // Choose a non-malicious block producer to be RPC so we can make sure it is
    // not getting forked off.
    let rpc_addr = cluster
        .validators
        .iter()
        .find(|(_, v)| v.info.contact_info.pubkey() != cluster.entry_point_info.pubkey())
        .and_then(|(_, v)| v.info.contact_info.rpc())
        .unwrap();
    let client = RpcClient::new_socket(rpc_addr);

    // Wait for stake to warm up so that no single node has more than super
    // minority stake.
    let _ = client.wait_for_max_stake(CommitmentConfig::confirmed(), 33.0);

    assert_matches!(enable_duplicate_block_attack(&cluster), Ok(()));

    // Sleep for 30 seconds to allow for duplicate block network partition to
    // occur.
    sleep(Duration::from_secs(30));

    assert_matches!(disable_duplicate_block_attack(&cluster), Ok(()));

    // If we are still making roots, it means the duplicate attack was not able
    // to partition and stall the cluster.
    cluster.check_for_new_roots(
        1,
        "test_duplicate_block_leaf_node_delivery",
        SocketAddrSpace::Unspecified,
    );
}
