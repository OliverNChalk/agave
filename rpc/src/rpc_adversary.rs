use {
    crate::rpc::JsonRpcRequestProcessor,
    jsonrpc_core::{Error, Result},
    jsonrpc_derive::rpc,
    solana_adversary::{
        adversary_context,
        adversary_feature_set::{
            self, drop_turbine_votes, example, gossip_packet_flood, invalidate_leader_block,
            packet_drop_parameters, repair_packet_flood, repair_parameters, replay_stage_attack,
            send_duplicate_blocks, shred_receiver_address, AdversaryFeatureConfig,
        },
        gossip::GossipPacketFlood,
        repair::RepairPacketFlood,
        verify_peer_identifier,
    },
    solana_metrics::metrics::public_metrics_db,
};

#[rpc]
pub trait Adversary {
    type Metadata;

    #[rpc(meta, name = "getAdversarialStatus")]
    fn get_status(&self, meta: Self::Metadata) -> Result<Vec<AdversaryFeatureConfig>>;

    #[rpc(meta, name = "configureExample")]
    fn configure_example(
        &self,
        meta: Self::Metadata,
        config: example::AdversarialConfig,
    ) -> Result<()>;

    #[rpc(meta, name = "configureRepairPacketFlood")]
    fn configure_repair_packet_flood(
        &self,
        meta: Self::Metadata,
        config: repair_packet_flood::AdversarialConfig,
    ) -> Result<()>;

    #[rpc(meta, name = "configureRepairParameters")]
    fn configure_repair_parameters(
        &self,
        meta: Self::Metadata,
        config: repair_parameters::AdversarialConfig,
    ) -> Result<()>;

    #[rpc(meta, name = "configureSendDuplicateBlocks")]
    fn configure_send_duplicate_blocks(
        &self,
        meta: Self::Metadata,
        config: send_duplicate_blocks::AdversarialConfig,
    ) -> Result<()>;

    #[rpc(meta, name = "configureShredReceiverAddress")]
    fn configure_shred_receiver_address(
        &self,
        meta: Self::Metadata,
        config: shred_receiver_address::AdversarialConfig,
    ) -> Result<()>;

    #[rpc(meta, name = "turbineVotes")]
    fn configure_drop_turbine_votes(
        &self,
        meta: Self::Metadata,
        config: drop_turbine_votes::AdversarialConfig,
    ) -> Result<()>;

    #[rpc(meta, name = "configureInvalidateLeaderBlock")]
    fn configure_invalidate_leader_block(
        &self,
        meta: Self::Metadata,
        config: invalidate_leader_block::AdversarialConfig,
    ) -> Result<()>;

    #[rpc(meta, name = "configurePacketDropParameters")]
    fn configure_packet_drop_parameters(
        &self,
        meta: Self::Metadata,
        config: packet_drop_parameters::AdversarialConfig,
    ) -> Result<()>;

    #[rpc(meta, name = "configureReplayStageAttack")]
    fn configure_replay_stage_attack(
        &self,
        meta: Self::Metadata,
        config: replay_stage_attack::AdversarialConfig,
    ) -> Result<()>;

    #[rpc(meta, name = "configureGossipPacketFlood")]
    fn configure_gossip_packet_flood(
        &self,
        meta: Self::Metadata,
        config: gossip_packet_flood::AdversarialConfig,
    ) -> Result<()>;

    fn perform_configuration<F>(&self, meta: Self::Metadata, configuration: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>;
}

// Detects which adversarial attacks are active and outputs metrics for each attack
fn output_adversary_metrics(adversary_feature_configs: Vec<AdversaryFeatureConfig>) {
    match public_metrics_db() {
        Ok(false) => (),
        Ok(true) => {
            info!("Bypassing adversary metrics for public cluster database.");
            return;
        }
        Err(e) => {
            error!(
                "Bypassing adversary metrics for unknown cluster database. Failed to query \
                 metrics configuration: {e}."
            );
            return;
        }
    }

    let mut repair_packet_flood = false;
    let mut send_duplicate_blocks = false;
    let mut drop_turbine_votes = false;
    let mut invalidate_leader_block = false;
    let mut replay_stage_attack = false;
    let mut gossip_packet_flood = false;

    for adversary_feature_config in adversary_feature_configs {
        match adversary_feature_config {
            AdversaryFeatureConfig::RepairPacketFlood(config) => {
                if !config.configs.is_empty() {
                    repair_packet_flood = true;
                }
            }
            AdversaryFeatureConfig::SendDuplicateBlocks(config) => {
                if config.num_duplicate_validators > 0 {
                    send_duplicate_blocks = true;
                }
            }
            AdversaryFeatureConfig::DropTurbineVotes(config) => {
                if config.drop_turbine_votes {
                    drop_turbine_votes = true;
                }
            }
            AdversaryFeatureConfig::InvalidateLeaderBlock(config) => {
                if config.invalidation_kind.is_some() {
                    invalidate_leader_block = true;
                }
            }
            AdversaryFeatureConfig::ReplayStageAttack(config) => {
                replay_stage_attack = config.selected_attack.is_some();
            }
            AdversaryFeatureConfig::GossipPacketFlood(config) => {
                if !config.configs.is_empty() {
                    gossip_packet_flood = true;
                }
            }
            AdversaryFeatureConfig::Example(_)
            | AdversaryFeatureConfig::PacketDropParameters(_)
            | AdversaryFeatureConfig::RepairParameters(_)
            | AdversaryFeatureConfig::ShredReceiverAddress(_) => {}
        }
    }

    datapoint_info!(
        "adversary",
        ("repair_packet_flood", repair_packet_flood, i64),
        ("send_duplicate_blocks", send_duplicate_blocks, i64),
        ("drop_turbine_votes", drop_turbine_votes, i64),
        ("invalidate_leader_block", invalidate_leader_block, i64),
        ("replay_stage_attack", replay_stage_attack, i64),
        ("gossip_packet_flood", gossip_packet_flood, i64),
    );
}

pub struct AdversaryImpl;
impl Adversary for AdversaryImpl {
    type Metadata = JsonRpcRequestProcessor;

    fn get_status(&self, _meta: Self::Metadata) -> Result<Vec<AdversaryFeatureConfig>> {
        Ok(adversary_feature_set::get_adversary_feature_status())
    }

    fn configure_example(
        &self,
        _meta: Self::Metadata,
        config: example::AdversarialConfig,
    ) -> Result<()> {
        example::set_config(config);
        Ok(())
    }

    fn configure_repair_packet_flood(
        &self,
        meta: Self::Metadata,
        config: repair_packet_flood::AdversarialConfig,
    ) -> Result<()> {
        self.perform_configuration(meta.clone(), || {
            for config in &config.configs {
                if let Some(target) = &config.target {
                    verify_peer_identifier(target)
                        .map_err(|e| Error::invalid_params(format!("Invalid param: {e}.")))?;
                }
            }
            let mut adversary_repair = adversary_context::ADVERSARY_CONTEXT
                .repair_packet_flood
                .write()
                .unwrap();
            if let Some(context) = adversary_repair.take() {
                context.join().unwrap();
            }
            repair_packet_flood::set_config(config.clone());
            if !config.configs.is_empty() {
                *adversary_repair = Some(RepairPacketFlood::start(
                    meta.serve_repair_socket(),
                    meta.bank_forks(),
                    meta.cluster_info(),
                    meta.leader_schedule_cache(),
                    config.configs,
                ));
            }
            Ok(())
        })
    }

    fn configure_repair_parameters(
        &self,
        meta: Self::Metadata,
        config: repair_parameters::AdversarialConfig,
    ) -> Result<()> {
        self.perform_configuration(meta, || {
            repair_parameters::set_config(config);
            Ok(())
        })
    }

    fn configure_send_duplicate_blocks(
        &self,
        meta: Self::Metadata,
        config: send_duplicate_blocks::AdversarialConfig,
    ) -> Result<()> {
        self.perform_configuration(meta, || {
            send_duplicate_blocks::set_config(config);
            Ok(())
        })
    }

    fn configure_shred_receiver_address(
        &self,
        meta: Self::Metadata,
        config: shred_receiver_address::AdversarialConfig,
    ) -> Result<()> {
        self.perform_configuration(meta, || {
            shred_receiver_address::set_config(config);
            Ok(())
        })
    }

    fn configure_drop_turbine_votes(
        &self,
        meta: Self::Metadata,
        config: drop_turbine_votes::AdversarialConfig,
    ) -> Result<()> {
        self.perform_configuration(meta, || {
            drop_turbine_votes::set_config(config);
            Ok(())
        })
    }

    fn configure_invalidate_leader_block(
        &self,
        meta: Self::Metadata,
        config: invalidate_leader_block::AdversarialConfig,
    ) -> Result<()> {
        self.perform_configuration(meta, || {
            invalidate_leader_block::set_config(config);
            Ok(())
        })
    }

    fn configure_packet_drop_parameters(
        &self,
        meta: Self::Metadata,
        config: packet_drop_parameters::AdversarialConfig,
    ) -> Result<()> {
        self.perform_configuration(meta, || {
            packet_drop_parameters::set_config(config);
            Ok(())
        })
    }

    fn configure_replay_stage_attack(
        &self,
        meta: Self::Metadata,
        config: replay_stage_attack::AdversarialConfig,
    ) -> Result<()> {
        self.perform_configuration(meta, || {
            replay_stage_attack::set_config(config);
            Ok(())
        })
    }

    fn configure_gossip_packet_flood(
        &self,
        meta: Self::Metadata,
        config: gossip_packet_flood::AdversarialConfig,
    ) -> Result<()> {
        self.perform_configuration(meta.clone(), || {
            for config in &config.configs {
                if let Some(target) = &config.target {
                    verify_peer_identifier(target)
                        .map_err(|e| Error::invalid_params(format!("Invalid param: {e}")))?;
                }
            }
            let mut adversary_repair = adversary_context::ADVERSARY_CONTEXT
                .gossip_packet_flood
                .write()
                .unwrap();
            if let Some(context) = adversary_repair.take() {
                context.join().unwrap();
            }
            gossip_packet_flood::set_config(config.clone());
            if !config.configs.is_empty() {
                *adversary_repair = Some(GossipPacketFlood::start(
                    meta.cluster_info(),
                    config.configs,
                ));
            }
            Ok(())
        })
    }

    fn perform_configuration<F>(&self, meta: Self::Metadata, configuration: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        output_adversary_metrics(self.get_status(meta.clone())?);
        configuration()?;
        output_adversary_metrics(self.get_status(meta)?);
        Ok(())
    }
}

#[cfg(test)]
pub mod tests {
    use {
        super::*,
        crate::rpc::tests::{create_test_request, parse_success_result},
        jsonrpc_core::{MetaIoHandler, Response, Value},
        serial_test::serial,
        solana_adversary::{
            adversary_context::ADVERSARY_CONTEXT,
            adversary_feature_set::{
                repair_packet_flood::{FloodConfig, FloodStrategy},
                replay_stage_attack,
            },
        },
        solana_ledger::genesis_utils::create_genesis_config,
        solana_runtime::bank::Bank,
        solana_send_transaction_service::{
            tpu_info::NullTpuInfo, transaction_client::ConnectionCacheClient,
        },
        solana_streamer::socket::SocketAddrSpace,
        std::{net::SocketAddr, sync::Arc},
    };

    fn setup_test_meta() -> JsonRpcRequestProcessor {
        let genesis = create_genesis_config(100);
        let bank = Bank::new_for_tests(&genesis.genesis_config);
        JsonRpcRequestProcessor::new_from_bank::<ConnectionCacheClient<NullTpuInfo>>(
            bank,
            SocketAddrSpace::Unspecified,
        )
    }

    fn handle_request_sync(
        io: &MetaIoHandler<JsonRpcRequestProcessor>,
        meta: JsonRpcRequestProcessor,
        request: serde_json::Value,
    ) -> Response {
        let response = io
            .handle_request_sync(&request.to_string(), meta)
            .expect("no response from handle_request_synce()");
        serde_json::from_str(&response).expect("failed to deserialize response")
    }

    #[test]
    #[serial]
    fn test_adversary_get_status() {
        let meta = setup_test_meta();
        let mut io = MetaIoHandler::default();
        io.extend_with(AdversaryImpl.to_delegate());

        let expected_result = json!(
            [{
                "turbineVotes": {
                    "dropTurbineVotes": false,
                },
            },
            {
                "exampleAdversarialConfig": {
                    "exampleNum": 0,
                },
            },
            {
                "gossipPacketFloodAdversarialConfig": {
                    "configs": [],
                },
            },
            {
                "invalidateLeaderBlockConfig": {
                    "invalidationKind": null,
                },
            },
            {
                "packetDropParametersConfig": {
                    "broadcastPacketDropPercent": null,
                    "retransmitPacketDropPercent": null,
                },
            },
            {
                "repairPacketFloodAdversarialConfig": {
                    "configs": [],
                },
            },
            {
                "repairParametersConfig": {
                    "serveRepairMaxRequestsPerIteration": null,
                    "serveRepairOversampledRequestsPerIteration": null,
                    "serveRepairAncestorHashesInvalidRespones": null,
                    "ancestorHashRepairSampleSize": null,
                },
            },
            {
                "replayStageAttack": {
                    "selectedAttack": null
                }
            },
            {
                "sendDuplicateBlocksConfig": {
                    "numDuplicateValidators": 0,
                    "newEntryIndexFromEnd": 0,
                    "sendOriginalAfterMs": 0,
                    "sendDestinations": [],
                },
            },
            {
                "shredReceiverAddress": {
                    "shredReceiverAddress": null,
                },
            }]
        );
        let request = create_test_request("getAdversarialStatus", None);
        let result: Value = parse_success_result(handle_request_sync(&io, meta, request));
        assert_eq!(result, expected_result);
    }

    #[test]
    #[serial]
    fn test_adversary_configure_example() {
        let meta = setup_test_meta();
        let mut io = MetaIoHandler::default();
        io.extend_with(AdversaryImpl.to_delegate());

        let config = example::AdversarialConfig { example_num: 313 };

        // Update the config for example, ensuring that request succeeds
        let request = create_test_request("configureExample", Some(json!([config])));
        let result: Value = parse_success_result(handle_request_sync(&io, meta.clone(), request));
        assert_eq!(result, json!(null));

        // Confirm that the config update is reflected internally
        assert_eq!(config, example::get_config());

        // Reset the config back to the default, to make sure other tests are not affected by the
        // state change.
        let default_config = example::AdversarialConfig::default();
        let request = create_test_request("configureExample", Some(json!([default_config])));
        let result: Value = parse_success_result(handle_request_sync(&io, meta, request));
        assert_eq!(result, json!(null));

        assert_eq!(default_config, example::get_config());
    }

    #[test]
    #[serial]
    fn test_adversary_configure_repair_packet_flood() {
        let meta = setup_test_meta();
        let mut io = MetaIoHandler::default();
        io.extend_with(AdversaryImpl.to_delegate());

        let config = repair_packet_flood::AdversarialConfig {
            configs: vec![FloodConfig {
                flood_strategy: FloodStrategy::MinimalPackets,
                packets_per_peer_per_iteration: 1,
                iteration_delay_us: 1_000_000,
                target: Some("9h1HyLCW5dZnBVap8C5egQ9Z6pHyjsh5MNy83iPqqRuq".to_string()),
            }],
        };

        {
            // Update the config for example, ensuring that request succeeds
            let meta = meta.clone();
            let request = create_test_request("configureRepairPacketFlood", Some(json!([config])));
            let result: Value = parse_success_result(handle_request_sync(&io, meta, request));
            assert_eq!(result, json!(null));
        }
        {
            let adversary_repair = ADVERSARY_CONTEXT.repair_packet_flood.read().unwrap();
            assert!(adversary_repair.is_some());
        }

        // Confirm that the config update is reflected internally
        assert_eq!(config, repair_packet_flood::get_config());

        // Reset the config
        let config = repair_packet_flood::AdversarialConfig::default();
        let request = create_test_request("configureRepairPacketFlood", Some(json!([config])));
        let result: Value = parse_success_result(handle_request_sync(&io, meta, request));
        assert_eq!(result, json!(null));
        {
            let adversary_repair = ADVERSARY_CONTEXT.repair_packet_flood.read().unwrap();
            assert!(adversary_repair.is_none());
        }
    }

    #[test]
    fn test_adversary_repair_packet_flood_decode() {
        let encoded_config = json!(repair_packet_flood::AdversarialConfig {
            configs: vec![FloodConfig {
                flood_strategy: FloodStrategy::MinimalPackets,
                packets_per_peer_per_iteration: 123,
                iteration_delay_us: 456,
                target: Some("10.0.0.9".to_string()),
            }],
        });
        let expected_config = json!({
            "configs": [{
                "floodStrategy": "minimalPackets",
                "iterationDelayUs": 456,
                "packetsPerPeerPerIteration": 123,
                "target": "10.0.0.9",
            }]
        });
        assert_eq!(encoded_config, expected_config);

        let encoded_config = json!(repair_packet_flood::AdversarialConfig {
            configs: vec![FloodConfig {
                flood_strategy: FloodStrategy::SignedPackets,
                packets_per_peer_per_iteration: 123,
                iteration_delay_us: 456,
                target: Some("::1".to_string()),
            }],
        });
        let expected_config = json!({
            "configs": [{
                "floodStrategy": "signedPackets",
                "iterationDelayUs": 456,
                "packetsPerPeerPerIteration": 123,
                "target": "::1",
            }]
        });
        assert_eq!(encoded_config, expected_config);

        let pubkey_str = "9h1HyLCW5dZnBVap8C5egQ9Z6pHyjsh5MNy83iPqqRuq";
        let encoded_config = json!(repair_packet_flood::AdversarialConfig {
            configs: vec![FloodConfig {
                flood_strategy: FloodStrategy::MinimalPackets,
                packets_per_peer_per_iteration: 123,
                iteration_delay_us: 456,
                target: Some(pubkey_str.to_string()),
            }],
        });
        let expected_config = json!({
            "configs": [{
                "floodStrategy": "minimalPackets",
                "iterationDelayUs": 456,
                "packetsPerPeerPerIteration": 123,
                "target": "9h1HyLCW5dZnBVap8C5egQ9Z6pHyjsh5MNy83iPqqRuq",
            }]
        });
        assert_eq!(encoded_config, expected_config);
    }

    #[test]
    #[serial]
    fn test_adversary_configure_shred_receiver_address() {
        let meta = setup_test_meta();
        let mut io = MetaIoHandler::default();
        io.extend_with(AdversaryImpl.to_delegate());

        let config = shred_receiver_address::AdversarialConfig {
            shred_receiver_address: Some("127.0.0.1:8080".parse().unwrap()),
        };

        {
            // Update the config, ensuring that request succeeds
            let meta = meta.clone();
            let request =
                create_test_request("configureShredReceiverAddress", Some(json!([config])));
            let result: Value = parse_success_result(handle_request_sync(&io, meta, request));
            assert_eq!(result, json!(null));
        }

        // Confirm that the config update is reflected internally
        assert_eq!(config, shred_receiver_address::get_config());

        // Reset the config
        let config = shred_receiver_address::AdversarialConfig::default();
        let request = create_test_request("configureShredReceiverAddress", Some(json!([config])));
        let result: Value = parse_success_result(handle_request_sync(&io, meta, request));
        assert_eq!(result, json!(null));

        // Confirm that the config update is reflected internally
        assert_eq!(config, shred_receiver_address::get_config());
    }

    #[test]
    #[serial]
    fn test_adversary_configure_send_duplicate_blocks() {
        let meta = setup_test_meta();
        let mut io = MetaIoHandler::default();
        io.extend_with(AdversaryImpl.to_delegate());

        let config = send_duplicate_blocks::AdversarialConfig {
            num_duplicate_validators: 2,
            new_entry_index_from_end: 1,
            send_original_after_ms: 500,
            send_destinations: vec![],
        };
        {
            // Update the config for send_duplicate_packets, ensuring that request succeeds
            let meta = meta.clone();
            let request =
                create_test_request("configureSendDuplicateBlocks", Some(json!([config])));
            let result: Value = parse_success_result(handle_request_sync(&io, meta, request));
            assert_eq!(result, json!(null));
        }
        // Confirm that the config update is reflected internally
        assert_eq!(config, send_duplicate_blocks::get_config());

        let config = send_duplicate_blocks::AdversarialConfig {
            num_duplicate_validators: 3,
            new_entry_index_from_end: 2,
            send_original_after_ms: 0,
            send_destinations: vec![
                Arc::new(vec![
                    SocketAddr::from(([127, 0, 0, 1], 234)),
                    SocketAddr::from(([10, 0, 0, 2], 345)),
                ]),
                Arc::new(vec![SocketAddr::from(([0x2023, 0, 0, 0, 0, 0, 0, 1], 987))]),
            ],
        };
        {
            // Update the config for send_duplicate_packets, ensuring that request succeeds
            let meta = meta.clone();
            let request =
                create_test_request("configureSendDuplicateBlocks", Some(json!([config])));
            let result: Value = parse_success_result(handle_request_sync(&io, meta, request));
            assert_eq!(result, json!(null));
        }
        // Confirm that the config update is reflected internally
        assert_eq!(config, send_duplicate_blocks::get_config());

        // Reset the config
        let config = send_duplicate_blocks::AdversarialConfig::default();
        let request = create_test_request("configureSendDuplicateBlocks", Some(json!([config])));
        let result: Value = parse_success_result(handle_request_sync(&io, meta, request));
        assert_eq!(result, json!(null));
    }

    #[test]
    #[serial]
    fn test_adversary_configure_drop_turbine_votes() {
        let meta = setup_test_meta();
        let mut io = MetaIoHandler::default();
        io.extend_with(AdversaryImpl.to_delegate());
        let mut config = drop_turbine_votes::AdversarialConfig::default();
        assert!(!config.drop_turbine_votes);
        config.drop_turbine_votes = true;
        {
            // Update the config, ensuring that request succeeds
            let meta = meta.clone();
            let request = create_test_request("turbineVotes", Some(json!([config])));
            let result: Value = parse_success_result(handle_request_sync(&io, meta, request));
            assert_eq!(result, json!(null));
        }

        // Confirm that the config update is reflected internally
        assert_eq!(config, drop_turbine_votes::get_config());

        // Reset the config
        let config = drop_turbine_votes::AdversarialConfig::default();
        let request = create_test_request("turbineVotes", Some(json!([config])));
        let result: Value = parse_success_result(handle_request_sync(&io, meta, request));
        assert_eq!(result, json!(null));

        // Confirm that the config update is reflected internally
        assert_eq!(config, drop_turbine_votes::get_config());
    }

    #[test]
    #[serial]
    fn test_adversary_configure_invalidate_leader_block() {
        let meta = setup_test_meta();
        let mut io = MetaIoHandler::default();
        io.extend_with(AdversaryImpl.to_delegate());

        let config = invalidate_leader_block::AdversarialConfig {
            invalidation_kind: Some(invalidate_leader_block::InvalidationKind::InvalidFeePayer),
        };
        {
            // Update the config for invalidate_leader_block, ensuring that request succeeds
            let meta = meta.clone();
            let request =
                create_test_request("configureInvalidateLeaderBlock", Some(json!([config])));
            let result: Value = parse_success_result(handle_request_sync(&io, meta, request));
            assert_eq!(result, json!(null));
        }
        // Confirm that the config update is reflected internally
        assert_eq!(config, invalidate_leader_block::get_config());

        let config = invalidate_leader_block::AdversarialConfig {
            invalidation_kind: Some(invalidate_leader_block::InvalidationKind::InvalidSignature),
        };
        {
            // Update the config for invalidate_leader_block, ensuring that request succeeds
            let meta = meta.clone();
            let request =
                create_test_request("configureInvalidateLeaderBlock", Some(json!([config])));
            let result: Value = parse_success_result(handle_request_sync(&io, meta, request));
            assert_eq!(result, json!(null));
        }
        // Confirm that the config update is reflected internally
        assert_eq!(config, invalidate_leader_block::get_config());

        // Reset the config
        let config = invalidate_leader_block::AdversarialConfig::default();
        let request = create_test_request("configureInvalidateLeaderBlock", Some(json!([config])));
        let result: Value = parse_success_result(handle_request_sync(&io, meta, request));
        assert_eq!(result, json!(null));
    }

    #[test]
    #[serial]
    fn test_adversary_configure_packet_drop_parameters() {
        let meta = setup_test_meta();
        let mut io = MetaIoHandler::default();
        io.extend_with(AdversaryImpl.to_delegate());

        let config = packet_drop_parameters::AdversarialConfig {
            broadcast_packet_drop_percent: Some(10),
            retransmit_packet_drop_percent: None,
        };

        {
            // Update the config for network_parameters, ensuring that request succeeds
            let meta = meta.clone();
            let request =
                create_test_request("configurePacketDropParameters", Some(json!([config])));
            let result: Value = parse_success_result(handle_request_sync(&io, meta, request));
            assert_eq!(result, json!(null));
        }

        // Confirm that the config update is reflected internally
        assert_eq!(config, packet_drop_parameters::get_config());

        // Reset the config
        let config = packet_drop_parameters::AdversarialConfig::default();
        let request = create_test_request("configurePacketDropParameters", Some(json!([config])));
        let result: Value = parse_success_result(handle_request_sync(&io, meta, request));
        assert_eq!(result, json!(null));
    }

    #[test]
    #[serial]
    fn test_adversary_configure_replay_stage_attack() {
        let meta = setup_test_meta();
        let mut io = MetaIoHandler::default();
        io.extend_with(AdversaryImpl.to_delegate());

        let config = replay_stage_attack::AdversarialConfig {
            selected_attack: Some(replay_stage_attack::Attack::TransferRandom),
        };
        {
            // Update the config for invalidate_leader_block, ensuring that request succeeds
            let meta = meta.clone();
            let request = create_test_request("configureReplayStageAttack", Some(json!([config])));
            let result: Value = parse_success_result(handle_request_sync(&io, meta, request));
            assert_eq!(result, json!(null));
        }
        // Confirm that the config update is reflected internally
        assert_eq!(config, replay_stage_attack::get_config());

        // Reset the config
        let config = replay_stage_attack::AdversarialConfig::default();
        let request = create_test_request("configureReplayStageAttack", Some(json!([config])));
        let result: Value = parse_success_result(handle_request_sync(&io, meta, request));
        assert_eq!(result, json!(null));
    }
}
