use {
    crate::rpc::{verify_pubkey, JsonRpcRequestProcessor},
    jsonrpc_core::Result,
    jsonrpc_derive::rpc,
    solana_adversary::{
        adversary_context,
        adversary_feature_set::{
            self, example,
            repair_packet_flood::{self, PeerIdentifier},
            repair_parameters, shred_receiver_address, AdversaryFeatureConfig,
        },
        repair::RepairPacketFlood,
    },
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

    #[rpc(meta, name = "configureShredReceiverAddress")]
    fn configure_shred_receiver_address(
        &self,
        meta: Self::Metadata,
        config: shred_receiver_address::AdversarialConfig,
    ) -> Result<()>;
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
        for config in &config.configs {
            if let Some(PeerIdentifier::Pubkey(pubkey)) = &config.target {
                verify_pubkey(pubkey.as_str())?;
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
            *adversary_repair = Some(RepairPacketFlood::new(
                meta.serve_repair_socket(),
                meta.bank_forks(),
                meta.cluster_info(),
                config.configs,
            ));
        }
        Ok(())
    }

    fn configure_repair_parameters(
        &self,
        _meta: Self::Metadata,
        config: repair_parameters::AdversarialConfig,
    ) -> Result<()> {
        repair_parameters::set_config(config);
        Ok(())
    }

    fn configure_shred_receiver_address(
        &self,
        _meta: Self::Metadata,
        config: shred_receiver_address::AdversarialConfig,
    ) -> Result<()> {
        shred_receiver_address::set_config(config);
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
            adversary_feature_set::repair_packet_flood::{FloodConfig, FloodStrategy},
        },
        solana_ledger::genesis_utils::create_genesis_config,
        solana_runtime::bank::Bank,
        solana_send_transaction_service::{
            tpu_info::NullTpuInfo, transaction_client::ConnectionCacheClient,
        },
        solana_streamer::socket::SocketAddrSpace,
        std::net::IpAddr,
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
                "exampleAdversarialConfig": {
                    "exampleNum": 0,
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
                target: Some(PeerIdentifier::Pubkey(
                    "9h1HyLCW5dZnBVap8C5egQ9Z6pHyjsh5MNy83iPqqRuq".to_string(),
                )),
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
                target: Some(PeerIdentifier::Ip(IpAddr::V4("10.0.0.9".parse().unwrap()))),
            }],
        });
        let expected_config = json!({
            "configs": [{
                "floodStrategy": "minimalPackets",
                "iterationDelayUs": 456,
                "packetsPerPeerPerIteration": 123,
                "target": {"ip": "10.0.0.9"},
            }]
        });
        assert_eq!(encoded_config, expected_config);

        let encoded_config = json!(repair_packet_flood::AdversarialConfig {
            configs: vec![FloodConfig {
                flood_strategy: FloodStrategy::SignedPackets,
                packets_per_peer_per_iteration: 123,
                iteration_delay_us: 456,
                target: Some(PeerIdentifier::Ip(IpAddr::V6("::1".parse().unwrap()))),
            }],
        });
        let expected_config = json!({
            "configs": [{
                "floodStrategy": "signedPackets",
                "iterationDelayUs": 456,
                "packetsPerPeerPerIteration": 123,
                "target": {"ip": "::1"},
            }]
        });
        assert_eq!(encoded_config, expected_config);

        let pubkey_str = "9h1HyLCW5dZnBVap8C5egQ9Z6pHyjsh5MNy83iPqqRuq";
        let encoded_config = json!(repair_packet_flood::AdversarialConfig {
            configs: vec![FloodConfig {
                flood_strategy: FloodStrategy::MinimalPackets,
                packets_per_peer_per_iteration: 123,
                iteration_delay_us: 456,
                target: Some(PeerIdentifier::Pubkey(pubkey_str.to_string())),
            }],
        });
        let expected_config = json!({
            "configs": [{
                "floodStrategy": "minimalPackets",
                "iterationDelayUs": 456,
                "packetsPerPeerPerIteration": 123,
                "target": {"pubkey": "9h1HyLCW5dZnBVap8C5egQ9Z6pHyjsh5MNy83iPqqRuq"},
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
}
