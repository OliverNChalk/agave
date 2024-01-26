use {
    serial_test::serial,
    solana_adversary::adversary_feature_set::send_duplicate_blocks,
    solana_cluster_type::ClusterType,
    solana_commitment_config::CommitmentConfig,
    solana_epoch_schedule::MINIMUM_SLOTS_PER_EPOCH,
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

fn configure_duplicate_block(url: &str) -> Result<(), String> {
    use {
        reqwest::blocking::Client,
        serde::{Deserialize, Serialize},
    };
    #[derive(Debug, Serialize, Deserialize)]
    struct RpcRequest {
        jsonrpc: String,
        method: String,
        params: serde_json::Value,
        id: u64,
    }

    let config = send_duplicate_blocks::AdversarialConfig {
        num_duplicate_validators: 0,
        new_entry_index_from_end: 2,
        send_original_after_ms: 0,
        turbine_send_delay_ms: 0,
        send_destinations: vec![],
        leaf_node_partitions: Some(2),
    };

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

#[test]
#[serial]
#[ignore]
// This test is ignored because it fails. There are fixes being worked to resolve these vulnerabilities.
fn test_mainnet_beta_cluster_type_duplicate_block() {
    solana_logger::setup_with_default(RUST_LOG_FILTER);

    // Enough nodes to generate a multi-layer turbine tree while not increasing
    // execution time too much.
    let num_nodes = 4;
    let mut config =
        ClusterConfig::new_with_equal_stakes(num_nodes, DEFAULT_MINT_LAMPORTS, DEFAULT_NODE_STAKE);
    config.cluster_type = ClusterType::MainnetBeta;
    config.slots_per_epoch = MINIMUM_SLOTS_PER_EPOCH;
    config.stakers_slot_offset = MINIMUM_SLOTS_PER_EPOCH;
    let cluster = LocalCluster::new(&mut config, SocketAddrSpace::Unspecified);
    let cluster_nodes = discover_validators(
        &cluster.entry_point_info.gossip().unwrap(),
        num_nodes,
        cluster.entry_point_info.shred_version(),
        SocketAddrSpace::Unspecified,
    )
    .unwrap();
    assert_eq!(cluster_nodes.len(), num_nodes);

    // Wait for node stakes to activate.
    let client = RpcClient::new_socket(cluster.entry_point_info.rpc().unwrap());
    let start_time = std::time::Instant::now();
    loop {
        let slot = client
            .get_slot_with_commitment(CommitmentConfig::default())
            .unwrap();
        if slot > MINIMUM_SLOTS_PER_EPOCH * 6 {
            // Stake is activated.
            break;
        }
        sleep(Duration::from_secs(1));
        if start_time.elapsed().as_secs() > 120 {
            panic!("Failed to activate stakes");
        }
    }

    // Enable duplicate block attack where leader directly sends to turbine leaf nodes.
    assert!(configure_duplicate_block(&cluster_tests::get_rpc_url(&cluster)).is_ok());

    // Sleep for 1 minute to monitor if network partition occurs.
    sleep(Duration::from_secs(60));

    // Get the highest Finalized slot from the cluster.
    let slot = client
        .get_slot_with_commitment(CommitmentConfig::default())
        .unwrap();

    // Cluster that does not partition (or effectively resolves partition)
    // should be able to reach finalized slot height of ~330. Attempts at
    // partitioning occur around slot 200. Slot 265 was chosen to provide
    // healthy margin on either side.
    log::info!("Finalized slot: {slot}");
    assert!(slot > 265);
}
