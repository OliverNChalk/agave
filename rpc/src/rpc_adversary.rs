use {
    crate::rpc::JsonRpcRequestProcessor,
    jsonrpc_core::Result,
    jsonrpc_derive::rpc,
    solana_adversary::{
        adversary_context,
        adversary_feature_set::{
            self, example, repair_minimal_packet_flood, AdversaryFeatureConfig,
        },
        repair::RepairMinimalPacketFlood,
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

    #[rpc(meta, name = "configureRepairMinimalPacketFlood")]
    fn configure_repair_minimal_packet_flood(
        &self,
        meta: Self::Metadata,
        config: repair_minimal_packet_flood::AdversarialConfig,
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

    fn configure_repair_minimal_packet_flood(
        &self,
        meta: Self::Metadata,
        config: repair_minimal_packet_flood::AdversarialConfig,
    ) -> Result<()> {
        let enable = config.enable;
        let mut adversary_repair = adversary_context::ADVERSARY_CONTEXT
            .repair_minimal_packet_flood
            .write()
            .unwrap();
        repair_minimal_packet_flood::set_config(config);
        if enable {
            if adversary_repair.is_none() {
                *adversary_repair = Some(RepairMinimalPacketFlood::new(
                    meta.serve_repair_socket(),
                    meta.cluster_info(),
                ));
                meta.validator_exit()
                    .write()
                    .unwrap()
                    .register_exit(Box::new(move || {
                        let mut adversary_repair = adversary_context::ADVERSARY_CONTEXT
                            .repair_minimal_packet_flood
                            .write()
                            .unwrap();
                        repair_minimal_packet_flood::set_config(
                            repair_minimal_packet_flood::AdversarialConfig::default(),
                        );
                        if let Some(context) = adversary_repair.take() {
                            context.join().unwrap();
                        }
                    }));
            }
        } else if let Some(context) = adversary_repair.take() {
            context.join().unwrap();
        }
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
        solana_ledger::genesis_utils::create_genesis_config,
        solana_runtime::bank::Bank,
        solana_send_transaction_service::{
            tpu_info::NullTpuInfo, transaction_client::ConnectionCacheClient,
        },
        solana_streamer::socket::SocketAddrSpace,
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
                "repairMinimalPacketFloodAdversarialConfig": {
                    "enable": false,
                    "iterationDelayUs": 0,
                    "packetsPerPeerPerIteration": 0,
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
}
