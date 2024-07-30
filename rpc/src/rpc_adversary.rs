use {
    crate::{rpc::JsonRpcRequestProcessor, rpc_service::AuthenticationMiddleware},
    jsonrpc_core::{Error, MetaIoHandler, Result},
    jsonrpc_derive::rpc,
    log::warn,
    lru::LruCache,
    rand::{thread_rng, Rng},
    solana_adversary::{
        adversary_context,
        adversary_feature_set::{
            self, delay_votes, drop_turbine_votes, example, gossip_packet_flood,
            invalidate_leader_block, packet_drop_parameters, repair_packet_flood,
            repair_parameters, replay_stage_attack, send_duplicate_blocks, shred_receiver_address,
            tpu_packet_flood, AdversaryFeatureConfig,
        },
        auth::{AuthHeader, JsonRpcAuthToken},
        gossip::GossipPacketFlood,
        repair::RepairPacketFlood,
        tpu, verify_peer_identifier, SelectedReplayAttack,
    },
    solana_metrics::metrics::should_bypass_adversary_metrics,
    solana_pubkey::Pubkey,
    std::{
        collections::HashSet,
        time::{Duration, Instant},
    },
};

pub struct JsonRpcSecurityTokenState {
    pub timestamp: Instant,
    pub nonce_history: HashSet<u32>,
}

pub const JSON_RPC_SECURITY_NONCE_CAPACITY: usize = 1_000;
pub const JSON_RPC_SECURITY_TOKEN_VALIDITY_DURATION: Duration = Duration::from_secs(10 * 60);
pub const JSON_RPC_SECURITY_ACTIVE_TOKEN_CAPACITY: usize = 100;
pub const JSON_RPC_SECURITY_PENDING_TOKEN_CAPACITY: usize = 100;

pub struct JsonRpcSecurityContext {
    // my id
    id: Option<Pubkey>,
    // active tokens, will expire after X time
    active_tokens: LruCache<JsonRpcAuthToken, JsonRpcSecurityTokenState>,
    // tokens pending first use
    pending_tokens: LruCache<JsonRpcAuthToken, ()>,
}

impl JsonRpcSecurityContext {
    pub fn new(id: Option<Pubkey>) -> Self {
        JsonRpcSecurityContext {
            id,
            active_tokens: LruCache::new(JSON_RPC_SECURITY_ACTIVE_TOKEN_CAPACITY),
            pending_tokens: LruCache::new(JSON_RPC_SECURITY_PENDING_TOKEN_CAPACITY),
        }
    }

    fn check_token(&mut self, auth_header: &AuthHeader) -> Result<()> {
        if self.pending_tokens.pop(&auth_header.token()).is_some() {
            // issued token was used, populate token state
            let token_state = JsonRpcSecurityTokenState {
                timestamp: Instant::now(),
                nonce_history: HashSet::from([auth_header.nonce()]),
            };
            self.active_tokens.put(auth_header.token(), token_state);
        } else if let Some(state) = self.active_tokens.get_mut(&auth_header.token()) {
            let age = Instant::now().saturating_duration_since(state.timestamp);
            if age > JSON_RPC_SECURITY_TOKEN_VALIDITY_DURATION {
                self.active_tokens.pop(&auth_header.token());
                return Err(Error::invalid_params("token expired"));
            }
            if state.nonce_history.contains(&auth_header.nonce()) {
                return Err(Error::invalid_params("nonce invalid"));
            }
            state.nonce_history.insert(auth_header.nonce());
            if state.nonce_history.len() > JSON_RPC_SECURITY_NONCE_CAPACITY {
                // limit nonce history by forcing token expiry
                self.active_tokens.pop(&auth_header.token());
            }
        } else {
            return Err(Error::invalid_params("token invalid"));
        }
        Ok(())
    }

    pub(crate) fn check_request(
        &mut self,
        call: &str,
        auth_header: &AuthHeader,
        payload_bytes: &[u8],
    ) -> Result<()> {
        let Some(id) = self.id else {
            return Ok(());
        };
        auth_header
            .verify_signature_serialized_payload(&id, call, payload_bytes)
            .map_err(Error::invalid_params)?;
        self.check_token(auth_header)
    }

    pub fn id(&self) -> Option<Pubkey> {
        self.id
    }
}

pub(crate) fn adversary_unauthenticated_call(call: &str) -> bool {
    matches!(call, "getAdversarialStatus" | "fetchAuthToken")
}

#[rpc]
pub trait Adversary {
    type Metadata;

    #[rpc(meta, name = "getAdversarialStatus")]
    fn get_status(&self, meta: Self::Metadata) -> Result<Vec<AdversaryFeatureConfig>>;

    #[rpc(meta, name = "fetchAuthToken")]
    fn fetch_auth_token(&self, meta: Self::Metadata) -> Result<JsonRpcAuthToken>;

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

    #[rpc(meta, name = "delayVotes")]
    fn configure_delay_votes(
        &self,
        meta: Self::Metadata,
        config: delay_votes::AdversarialConfig,
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

    #[rpc(meta, name = "configureTpuPacketFlood")]
    fn configure_tpu_packet_flood(
        &self,
        meta: Self::Metadata,
        config: tpu_packet_flood::AdversarialConfig,
    ) -> Result<()>;

    fn perform_configuration<F>(&self, meta: Self::Metadata, configuration: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>;
}

// Detects which adversarial attacks are active and outputs metrics for each attack
fn output_adversary_metrics(adversary_feature_configs: Vec<AdversaryFeatureConfig>) {
    if should_bypass_adversary_metrics() {
        return;
    }

    let mut repair_packet_flood = repair_packet_flood::FloodStrategySubtypeStatsId::default();
    let mut send_duplicate_blocks = false;
    let mut send_duplicate_leaf_nodes = false;
    let mut broadcast_delay = false;
    let mut broadcast_packet_drop_percent = 0;
    let mut retransmit_packet_drop_percent = 0;
    let mut drop_turbine_votes = false;
    let mut invalidate_leader_block =
        invalidate_leader_block::InvalidationKindSubtypeStatsId::default();
    let mut replay_stage_attack = replay_stage_attack::AttackSubtypeStatsId::default();
    let mut gossip_packet_flood = gossip_packet_flood::FloodStrategySubtypeStatsId::default();
    let mut tpu_packet_flood = false;
    let mut delay_votes = false;

    for adversary_feature_config in adversary_feature_configs {
        match adversary_feature_config {
            AdversaryFeatureConfig::RepairPacketFlood(config) => {
                if let Some(config) = config.configs.first() {
                    repair_packet_flood = config.flood_strategy.stats_id();
                }

                if config.configs.len() > 1 {
                    warn!(
                        "More than one repair flood config found. Running multiple repair attacks \
                         concurrently is not supported. Only taking the first one."
                    );
                }
            }
            AdversaryFeatureConfig::SendDuplicateBlocks(config) => {
                if config.num_duplicate_validators > 0 {
                    send_duplicate_blocks = true;
                }

                if config.turbine_send_delay_ms > 0 {
                    broadcast_delay = true;
                }

                if config.leaf_node_partitions.is_some() {
                    send_duplicate_leaf_nodes = true;
                }
            }
            AdversaryFeatureConfig::DropTurbineVotes(config) => {
                if config.drop_turbine_votes {
                    drop_turbine_votes = true;
                }
            }
            AdversaryFeatureConfig::InvalidateLeaderBlock(config) => {
                if let Some(invalidation_kind) = config.invalidation_kind {
                    invalidate_leader_block = invalidation_kind.stats_id();
                }
            }
            AdversaryFeatureConfig::ReplayStageAttack(config) => {
                if let Some(attack) = config.selected_attack {
                    replay_stage_attack = attack.stats_id();
                }
            }
            AdversaryFeatureConfig::GossipPacketFlood(config) => {
                if let Some(config) = config.configs.first() {
                    gossip_packet_flood = config.flood_strategy.stats_id();
                }

                if config.configs.len() > 1 {
                    warn!(
                        "More than one gossip flood config found. Running multiple gossip attacks \
                         concurrently is not supported. Only taking the first one."
                    );
                }
            }
            AdversaryFeatureConfig::TpuPacketFlood(config) => {
                if !config.configs.is_empty() {
                    tpu_packet_flood = true;
                }
            }
            AdversaryFeatureConfig::DelayVotes(config) => {
                if config.delay_votes_by_slot_count > 0 {
                    delay_votes = true;
                }
            }
            AdversaryFeatureConfig::PacketDropParameters(config) => {
                broadcast_packet_drop_percent = config.broadcast_packet_drop_percent.unwrap_or(0);
                retransmit_packet_drop_percent = config.retransmit_packet_drop_percent.unwrap_or(0);
            }
            AdversaryFeatureConfig::Example(_)
            | AdversaryFeatureConfig::RepairParameters(_)
            | AdversaryFeatureConfig::ShredReceiverAddress(_) => {}
        }
    }

    datapoint_info!(
        "adversary",
        ("repair_packet_flood", i64::from(repair_packet_flood), i64),
        ("send_duplicate_blocks", send_duplicate_blocks, i64),
        ("send_duplicate_leaf_nodes", send_duplicate_leaf_nodes, i64),
        ("broadcast_delay", broadcast_delay, i64),
        (
            "broadcast_packet_drop_percent",
            broadcast_packet_drop_percent,
            i64
        ),
        (
            "retransmit_packet_drop_percent",
            retransmit_packet_drop_percent,
            i64
        ),
        ("drop_turbine_votes", drop_turbine_votes, i64),
        (
            "invalidate_leader_block",
            i64::from(invalidate_leader_block),
            i64
        ),
        ("replay_stage_attack", i64::from(replay_stage_attack), i64),
        ("gossip_packet_flood", i64::from(gossip_packet_flood), i64),
        ("tpu_packet_flood", tpu_packet_flood, i64),
        ("delay_votes", delay_votes, i64),
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

    fn fetch_auth_token(&self, meta: Self::Metadata) -> Result<JsonRpcAuthToken> {
        let mut rng = thread_rng();
        let mut token = JsonRpcAuthToken::default();
        rng.fill(&mut token);
        meta.adversary_meta()
            .security
            .lock()
            .unwrap()
            .pending_tokens
            .put(token, ());
        Ok(token)
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

    fn configure_delay_votes(
        &self,
        meta: Self::Metadata,
        config: delay_votes::AdversarialConfig,
    ) -> Result<()> {
        self.perform_configuration(meta, || {
            delay_votes::set_config(config);
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
        self.perform_configuration(meta.clone(), || {
            let selected_attack = match config.selected_attack {
                Some(ref selected_attack) => {
                    let accounts_manager = meta.block_generator_config();
                    let Some(accounts_manager) = accounts_manager else {
                        return Err(Error::invalid_params(
                            "Cannot launch attack: accounts were not setup".to_string(),
                        ));
                    };

                    let accounts = accounts_manager.get_accounts();
                    selected_attack
                        .verify(&accounts)
                        .map_err(|message| Error::invalid_params(message.to_string()))?;

                    SelectedReplayAttack::Selected {
                        attack: selected_attack.clone(),
                        accounts,
                    }
                }
                None => SelectedReplayAttack::None,
            };

            // We still need to use this configuration to drop received packets in banking stage during attack
            replay_stage_attack::set_config(config);

            if let Some(replay_attack_sender) = meta.replay_attack_sender() {
                replay_attack_sender
                    .send(selected_attack)
                    .unwrap_or_else(|err| warn!("Failed to send replay attack: {err}"));
            }
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

    fn configure_tpu_packet_flood(
        &self,
        meta: Self::Metadata,
        config: tpu_packet_flood::AdversarialConfig,
    ) -> Result<()> {
        self.perform_configuration(meta.clone(), || {
            let mut adversary_tpu = adversary_context::ADVERSARY_CONTEXT
                .tpu_packet_flood
                .write()
                .unwrap();
            if let Some(context) = adversary_tpu.take() {
                context.join().unwrap();
            }
            tpu_packet_flood::set_config(config.clone());
            if !config.configs.is_empty() {
                *adversary_tpu = Some(tpu::start(
                    meta.cluster_info(),
                    meta.bank_forks(),
                    meta.leader_schedule_cache(),
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

pub type MetaIoWithAuth = MetaIoHandler<JsonRpcRequestProcessor, AuthenticationMiddleware>;

#[cfg(feature = "dev-context-only-utils")]
pub mod test_helpers {
    use {
        super::*,
        crate::{
            rpc::{
                test_helpers::{create_test_request, parse_success_result},
                JsonRpcRequestProcessor,
            },
            rpc_service::AuthenticationMiddleware,
        },
        http::HeaderMap,
        jsonrpc_core::{MetaIoHandler, Response},
        serde_json::{json, Value},
        solana_adversary::auth::HTTP_HEADER_FIELD_NAME_INVALIDATOR_AUTH,
        solana_keypair::Keypair,
        solana_ledger::genesis_utils::create_genesis_config,
        solana_runtime::bank::Bank,
        solana_send_transaction_service::{
            tpu_info::NullTpuInfo, transaction_client::ConnectionCacheClient,
        },
        solana_signer::Signer,
        solana_streamer::socket::SocketAddrSpace,
    };

    pub fn setup_test_no_auth() -> (JsonRpcRequestProcessor, Keypair, MetaIoWithAuth) {
        let genesis = create_genesis_config(100);
        let bank = Bank::new_for_tests(&genesis.genesis_config);
        let meta = JsonRpcRequestProcessor::new_from_bank::<ConnectionCacheClient<NullTpuInfo>>(
            bank,
            SocketAddrSpace::Unspecified,
        );
        let keypair = Keypair::new();
        meta.adversary_meta().security.lock().unwrap().id = Some(keypair.pubkey());

        let middleware = AuthenticationMiddleware::new(
            AdversaryImpl
                .to_delegate()
                .into_iter()
                .map(|(name, _)| name)
                .filter(|name| !adversary_unauthenticated_call(&name[..]))
                .collect(),
        );
        let mut io = MetaIoHandler::with_middleware(middleware);
        io.extend_with(AdversaryImpl.to_delegate());

        (meta, keypair, io)
    }

    pub fn setup_test() -> (
        JsonRpcRequestProcessor,
        Keypair,
        MetaIoWithAuth,
        JsonRpcAuthToken,
    ) {
        let (meta, keypair, io) = setup_test_no_auth();
        let token = fetch_auth_token(meta.clone(), &io);
        (meta, keypair, io, token)
    }

    fn fetch_auth_token(meta: JsonRpcRequestProcessor, io: &MetaIoWithAuth) -> JsonRpcAuthToken {
        let request = create_test_request("fetchAuthToken", None);
        let result: Value = parse_success_result(handle_request_sync(io, meta, request));
        let token: JsonRpcAuthToken = serde_json::from_value(result).unwrap();
        token
    }

    pub fn handle_request_sync(
        io: &MetaIoWithAuth,
        meta: JsonRpcRequestProcessor,
        request: Value,
    ) -> Response {
        let response = io
            .handle_request_sync(&request.to_string(), meta)
            .expect("no response from handle_request_synce()");
        serde_json::from_str(&response).expect("failed to deserialize response")
    }

    pub fn send_signed_request_sync<T: serde::Serialize>(
        meta: JsonRpcRequestProcessor,
        io: &MetaIoWithAuth,
        keypair: &Keypair,
        call: &str,
        token: &JsonRpcAuthToken,
        payload: &T,
    ) -> Response {
        let auth_header = AuthHeader::new_signed(token, keypair, call, payload).unwrap();
        let header_value = auth_header.to_header_value().unwrap();
        let mut meta = meta.clone();
        let mut headers = HeaderMap::default();
        headers.insert(HTTP_HEADER_FIELD_NAME_INVALIDATOR_AUTH, header_value);
        meta.adversary_meta_mut().set_headers(Some(headers));
        let params = json!([payload]);
        let request = create_test_request(call, Some(params));
        handle_request_sync(io, meta, request)
    }
}

#[cfg(test)]
pub mod tests {
    use {
        super::{
            test_helpers::{
                handle_request_sync, send_signed_request_sync, setup_test, setup_test_no_auth,
            },
            *,
        },
        crate::rpc::test_helpers::{
            create_test_request, parse_failure_response, parse_success_result,
        },
        http::HeaderMap,
        serde_json::{json, Value},
        serial_test::serial,
        solana_adversary::{
            adversary_context::ADVERSARY_CONTEXT,
            adversary_feature_set::repair_packet_flood::{FloodConfig, FloodStrategy},
            auth::HTTP_HEADER_FIELD_NAME_INVALIDATOR_AUTH,
        },
        std::{net::SocketAddr, sync::Arc},
    };

    #[test]
    #[serial]
    fn test_adversary_get_status() {
        let (meta, _, io) = setup_test_no_auth();

        let expected_result = json!(
            [{
                "delayVotes": {
                    "delayVotesBySlotCount": 0
                },
            },
            {
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
                    "turbineSendDelayMs": 0,
                    "leafNodePartitions": null,
                },
            },
            {
                "shredReceiverAddress": {
                    "shredReceiverAddress": null,
                },
            },
            {
                "tpuPacketFlood": {
                    "configs": [],
                },
            }]
        );
        let request = create_test_request("getAdversarialStatus", None);
        let result: Value = parse_success_result(handle_request_sync(&io, meta, request));
        assert_eq!(result, expected_result);
    }

    #[test]
    #[serial]
    fn test_adversary_fetch_auth_token() {
        let (mut meta, keypair, io, token) = setup_test();

        let payload = repair_packet_flood::AdversarialConfig {
            configs: vec![FloodConfig {
                flood_strategy: FloodStrategy::MinimalPackets,
                packets_per_peer_per_iteration: 1,
                iteration_delay_us: 1_000_000,
                target: Some("9h1HyLCW5dZnBVap8C5egQ9Z6pHyjsh5MNy83iPqqRuq".to_string()),
            }],
        };
        let mut auth_header =
            AuthHeader::new_signed(&token, &keypair, "configureRepairPacketFlood", &payload)
                .unwrap();
        let header_value = auth_header.to_header_value().unwrap();
        let mut headers = HeaderMap::default();
        headers.insert(HTTP_HEADER_FIELD_NAME_INVALIDATOR_AUTH, header_value);
        meta.adversary_meta_mut().set_headers(Some(headers));

        //let params = json!([auth_header, payload]);
        let params = json!([payload]);
        let request = create_test_request("configureRepairPacketFlood", Some(params));
        parse_success_result::<Value>(handle_request_sync(&io, meta.clone(), request));

        auth_header
            .sign_request(&keypair, "configureRepairPacketFlood", &payload)
            .unwrap();
        let header_value = auth_header.to_header_value().unwrap();
        let mut headers = HeaderMap::default();
        headers.insert(HTTP_HEADER_FIELD_NAME_INVALIDATOR_AUTH, header_value);
        meta.adversary_meta_mut().set_headers(Some(headers));

        let params = json!([payload]);
        let request = create_test_request("configureRepairPacketFlood", Some(params));
        let (_err, msg) = parse_failure_response(handle_request_sync(&io, meta.clone(), request));
        assert_eq!(msg, "Invalid params: nonce invalid");

        // Reset the config
        let config = repair_packet_flood::AdversarialConfig::default();
        let rsp = send_signed_request_sync(
            meta,
            &io,
            &keypair,
            "configureRepairPacketFlood",
            &token,
            &config,
        );
        let result: Value = parse_success_result(rsp);
        assert_eq!(result, json!(null));
        {
            let adversary_repair = ADVERSARY_CONTEXT.repair_packet_flood.read().unwrap();
            assert!(adversary_repair.is_none());
        }
    }

    #[test]
    #[serial]
    fn test_adversary_configure_example() {
        let (meta, keypair, io, token) = setup_test();

        // Update the config for example, ensuring that request succeeds
        let config = example::AdversarialConfig { example_num: 313 };
        let rsp = send_signed_request_sync(
            meta.clone(),
            &io,
            &keypair,
            "configureExample",
            &token,
            &config,
        );
        let result: Value = parse_success_result(rsp);
        assert_eq!(result, json!(null));

        // Confirm that the config update is reflected internally
        assert_eq!(config, example::get_config());

        // Reset the config back to the default, to make sure other tests are not affected by the
        // state change.
        let default_config = example::AdversarialConfig::default();
        let rsp = send_signed_request_sync(
            meta,
            &io,
            &keypair,
            "configureExample",
            &token,
            &default_config,
        );
        let result: Value = parse_success_result(rsp);
        assert_eq!(result, json!(null));

        assert_eq!(default_config, example::get_config());
    }

    #[test]
    #[serial]
    fn test_adversary_configure_repair_packet_flood() {
        let (meta, keypair, io, token) = setup_test();

        // Update the config for example, ensuring that request succeeds
        let config = repair_packet_flood::AdversarialConfig {
            configs: vec![FloodConfig {
                flood_strategy: FloodStrategy::MinimalPackets,
                packets_per_peer_per_iteration: 1,
                iteration_delay_us: 1_000_000,
                target: Some("9h1HyLCW5dZnBVap8C5egQ9Z6pHyjsh5MNy83iPqqRuq".to_string()),
            }],
        };
        let rsp = send_signed_request_sync(
            meta.clone(),
            &io,
            &keypair,
            "configureRepairPacketFlood",
            &token,
            &config,
        );
        let result: Value = parse_success_result(rsp);
        assert_eq!(result, json!(null));

        {
            let adversary_repair = ADVERSARY_CONTEXT.repair_packet_flood.read().unwrap();
            assert!(adversary_repair.is_some());
        }

        // Confirm that the config update is reflected internally
        assert_eq!(config, repair_packet_flood::get_config());

        // Reset the config
        let config = repair_packet_flood::AdversarialConfig::default();
        let rsp = send_signed_request_sync(
            meta,
            &io,
            &keypair,
            "configureRepairPacketFlood",
            &token,
            &config,
        );
        let result: Value = parse_success_result(rsp);
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
        let (meta, keypair, io, token) = setup_test();

        // Update the config, ensuring that request succeeds
        let config = shred_receiver_address::AdversarialConfig {
            shred_receiver_address: Some("127.0.0.1:8080".parse().unwrap()),
        };
        let rsp = send_signed_request_sync(
            meta.clone(),
            &io,
            &keypair,
            "configureShredReceiverAddress",
            &token,
            &config,
        );
        let result: Value = parse_success_result(rsp);
        assert_eq!(result, json!(null));

        // Confirm that the config update is reflected internally
        assert_eq!(config, shred_receiver_address::get_config());

        // Reset the config
        let config = shred_receiver_address::AdversarialConfig::default();
        let rsp = send_signed_request_sync(
            meta,
            &io,
            &keypair,
            "configureShredReceiverAddress",
            &token,
            &config,
        );
        let result: Value = parse_success_result(rsp);
        assert_eq!(result, json!(null));

        // Confirm that the config update is reflected internally
        assert_eq!(config, shred_receiver_address::get_config());
    }

    #[test]
    #[serial]
    fn test_adversary_configure_send_duplicate_blocks() {
        let (meta, keypair, io, token) = setup_test();

        // Update the config for send_duplicate_packets, ensuring that request succeeds
        let config = send_duplicate_blocks::AdversarialConfig {
            num_duplicate_validators: 2,
            new_entry_index_from_end: 1,
            send_original_after_ms: 500,
            turbine_send_delay_ms: 0,
            send_destinations: vec![],
            leaf_node_partitions: None,
        };
        let rsp = send_signed_request_sync(
            meta.clone(),
            &io,
            &keypair,
            "configureSendDuplicateBlocks",
            &token,
            &config,
        );
        let result: Value = parse_success_result(rsp);
        assert_eq!(result, json!(null));

        // Confirm that the config update is reflected internally
        assert_eq!(config, send_duplicate_blocks::get_config());

        // Update the config for send_duplicate_packets, ensuring that request succeeds
        let config = send_duplicate_blocks::AdversarialConfig {
            num_duplicate_validators: 3,
            new_entry_index_from_end: 2,
            send_original_after_ms: 0,
            turbine_send_delay_ms: 0,
            send_destinations: vec![
                Arc::new(vec![
                    SocketAddr::from(([127, 0, 0, 1], 234)),
                    SocketAddr::from(([10, 0, 0, 2], 345)),
                ]),
                Arc::new(vec![SocketAddr::from(([0x2023, 0, 0, 0, 0, 0, 0, 1], 987))]),
            ],
            leaf_node_partitions: None,
        };
        let rsp = send_signed_request_sync(
            meta.clone(),
            &io,
            &keypair,
            "configureSendDuplicateBlocks",
            &token,
            &config,
        );
        let result: Value = parse_success_result(rsp);
        assert_eq!(result, json!(null));

        // Confirm that the config update is reflected internally
        assert_eq!(config, send_duplicate_blocks::get_config());

        // Reset the config
        let config = send_duplicate_blocks::AdversarialConfig::default();
        let rsp = send_signed_request_sync(
            meta,
            &io,
            &keypair,
            "configureSendDuplicateBlocks",
            &token,
            &config,
        );
        let result: Value = parse_success_result(rsp);
        assert_eq!(result, json!(null));
    }

    #[test]
    #[serial]
    fn test_adversary_configure_drop_turbine_votes() {
        let (meta, keypair, io, token) = setup_test();

        // Update the config, ensuring that request succeeds
        let mut config = drop_turbine_votes::AdversarialConfig::default();
        assert!(!config.drop_turbine_votes);
        config.drop_turbine_votes = true;
        let rsp =
            send_signed_request_sync(meta.clone(), &io, &keypair, "turbineVotes", &token, &config);
        let result: Value = parse_success_result(rsp);
        assert_eq!(result, json!(null));

        // Confirm that the config update is reflected internally
        assert_eq!(config, drop_turbine_votes::get_config());

        // Reset the config
        let config = drop_turbine_votes::AdversarialConfig::default();
        let rsp = send_signed_request_sync(meta, &io, &keypair, "turbineVotes", &token, &config);
        let result: Value = parse_success_result(rsp);
        assert_eq!(result, json!(null));

        // Confirm that the config update is reflected internally
        assert_eq!(config, drop_turbine_votes::get_config());
    }

    #[test]
    #[serial]
    fn test_adversary_configure_invalidate_leader_block() {
        let (meta, keypair, io, token) = setup_test();

        // Update the config for invalidate_leader_block, ensuring that request succeeds
        let config = invalidate_leader_block::AdversarialConfig {
            invalidation_kind: Some(invalidate_leader_block::InvalidationKind::InvalidFeePayer),
        };
        let rsp = send_signed_request_sync(
            meta.clone(),
            &io,
            &keypair,
            "configureInvalidateLeaderBlock",
            &token,
            &config,
        );
        let result: Value = parse_success_result(rsp);
        assert_eq!(result, json!(null));

        // Confirm that the config update is reflected internally
        assert_eq!(config, invalidate_leader_block::get_config());

        // Update the config for invalidate_leader_block, ensuring that request succeeds
        let config = invalidate_leader_block::AdversarialConfig {
            invalidation_kind: Some(invalidate_leader_block::InvalidationKind::InvalidSignature),
        };
        let rsp = send_signed_request_sync(
            meta.clone(),
            &io,
            &keypair,
            "configureInvalidateLeaderBlock",
            &token,
            &config,
        );
        let result: Value = parse_success_result(rsp);
        assert_eq!(result, json!(null));

        // Confirm that the config update is reflected internally
        assert_eq!(config, invalidate_leader_block::get_config());

        // Reset the config
        let config = invalidate_leader_block::AdversarialConfig::default();
        let rsp = send_signed_request_sync(
            meta,
            &io,
            &keypair,
            "configureInvalidateLeaderBlock",
            &token,
            &config,
        );
        let result: Value = parse_success_result(rsp);
        assert_eq!(result, json!(null));
    }

    #[test]
    #[serial]
    fn test_adversary_configure_packet_drop_parameters() {
        let (meta, keypair, io, token) = setup_test();

        // Update the config for network_parameters, ensuring that request succeeds
        let config = packet_drop_parameters::AdversarialConfig {
            broadcast_packet_drop_percent: Some(10),
            retransmit_packet_drop_percent: None,
        };
        let rsp = send_signed_request_sync(
            meta.clone(),
            &io,
            &keypair,
            "configurePacketDropParameters",
            &token,
            &config,
        );
        let result: Value = parse_success_result(rsp);
        assert_eq!(result, json!(null));

        // Confirm that the config update is reflected internally
        assert_eq!(config, packet_drop_parameters::get_config());

        // Reset the config
        let config = packet_drop_parameters::AdversarialConfig::default();
        let rsp = send_signed_request_sync(
            meta,
            &io,
            &keypair,
            "configurePacketDropParameters",
            &token,
            &config,
        );
        let result: Value = parse_success_result(rsp);
        assert_eq!(result, json!(null));
    }
}
