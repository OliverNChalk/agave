use {
    crossbeam_channel::Receiver,
    indoc::formatdoc,
    log::{debug, error, info},
    serial_test::serial,
    solana_account::{Account, AccountSharedData},
    solana_adversary::{
        accounts_file::AccountsFile,
        adversary_feature_set::replay_stage_attack::{
            self, Attack, AttackProgramConfig, LargeNopAttackConfig,
            NonExistentAccountsAttackConfig,
        },
        block_generator_config::{BlockGeneratorAccountsSource, BlockGeneratorConfig},
        send_request_verified,
    },
    solana_client::{
        pubsub_client::PubsubClientSubscription, rpc_client::RpcClient,
        rpc_response::RpcBlockUpdate,
    },
    solana_clock::DEFAULT_SLOTS_PER_EPOCH,
    solana_cluster_type::ClusterType,
    solana_commitment_config::CommitmentConfig,
    solana_compute_budget::compute_budget_limits::MAX_COMPUTE_UNIT_LIMIT,
    solana_core::{
        banking_stage::adversary::generator_templates::{
            max_size_accounts_tx::TX_MAX_NUM_MAX_SIZE_ACCOUNTS, rotate_accounts::BATCH_SIZE,
        },
        validator::{InvalidatorConfig, ValidatorConfig},
    },
    solana_gossip::gossip_service::discover_validators,
    solana_keypair::Keypair,
    solana_ledger::leader_schedule::FixedSchedule,
    solana_local_cluster::{
        cluster::Cluster,
        cluster_tests,
        integration_tests::{create_custom_leader_schedule, DEFAULT_NODE_STAKE, RUST_LOG_FILTER},
        local_cluster::{ClusterConfig, LocalCluster},
        validator_configs::*,
    },
    solana_pubkey::Pubkey,
    solana_pubsub_client::pubsub_client::PubsubClient,
    solana_rent::Rent,
    solana_rpc_client_api::{
        config::{RpcBlockSubscribeConfig, RpcBlockSubscribeFilter},
        response::Response,
    },
    solana_sdk_ids::{bpf_loader, system_program},
    solana_signer::Signer,
    solana_stake_interface::stake_history::Epoch,
    solana_streamer::socket::SocketAddrSpace,
    solana_transaction_status::{TransactionDetails, UiConfirmedBlock},
    std::{iter, sync::Arc, thread::sleep, time::Duration},
    tempfile::tempdir,
};

const NUM_NODES: usize = 2;
const STRESS_TEST_PROGRAM_ID: Pubkey = Pubkey::new_from_array([7u8; 32]);
const BLOCK_VALIDATION_COMMITMENT_LEVEL: Option<CommitmentConfig> =
    Some(CommitmentConfig::confirmed());

mod setup {
    use super::*;

    /// Sets up a local cluster instance.
    pub fn create_cluster(
        accounts_file: Arc<AccountsFile>,
        starting_accounts: Vec<(Pubkey, AccountSharedData)>,
    ) -> LocalCluster {
        // Create custom leader schedule where bootstrap node gets all the
        // leader slots so it gets every opportunity to generate malicious
        // blocks.
        let validator_keys: Option<Vec<(Arc<Keypair>, bool)>> = Some(
            (0..NUM_NODES)
                .map(|_| (Arc::new(Keypair::new()), true))
                .collect(),
        );
        let bootstrap_node_pubkey = validator_keys.as_ref().unwrap().first().unwrap().0.pubkey();
        let validator_to_slots = vec![(bootstrap_node_pubkey, DEFAULT_SLOTS_PER_EPOCH as usize)];
        let leader_schedule = create_custom_leader_schedule(validator_to_slots.into_iter());

        // Create validator config with starting accounts in genesis that can be
        // used for generating malicious blocks.
        let mut validator_config = ValidatorConfig {
            invalidator_config: InvalidatorConfig {
                block_generator_config: Some(BlockGeneratorConfig {
                    accounts: BlockGeneratorAccountsSource::Genesis(accounts_file),
                }),
                rpc_adversary_id: None,
            },
            fixed_leader_schedule: Some(FixedSchedule {
                leader_schedule: Arc::new(leader_schedule),
            }),
            ..ValidatorConfig::default_for_test()
        };
        validator_config.enable_default_rpc_block_subscribe();

        // Create a cluster config with the generator validator config and equal
        // stake split so that any consensus divergence will be caught by virtue
        // of not making roots.
        let mut config = ClusterConfig {
            cluster_type: ClusterType::Development,
            node_stakes: vec![DEFAULT_NODE_STAKE; NUM_NODES],
            validator_configs: make_identical_validator_configs(&validator_config, NUM_NODES),
            validator_keys,
            additional_accounts: starting_accounts,
            ..ClusterConfig::default()
        };

        // Create the cluster.
        let cluster = LocalCluster::new(&mut config, SocketAddrSpace::Unspecified);
        let cluster_nodes = discover_validators(
            &cluster.entry_point_info.gossip().unwrap(),
            NUM_NODES,
            cluster.entry_point_info.shred_version(),
            SocketAddrSpace::Unspecified,
        )
        .unwrap();
        assert_eq!(cluster_nodes.len(), NUM_NODES);

        cluster
    }

    pub fn block_subscriber(
        cluster: &LocalCluster,
    ) -> (
        PubsubClientSubscription<Response<RpcBlockUpdate>>,
        Receiver<Response<RpcBlockUpdate>>,
    ) {
        // It's important we subscribe to the node that will be replaying (as
        // opposed to building) malicious blocks due to the following:
        //
        // 1. Tests will panic if we are unable to fetch block updates because
        //    we need them to verify test behavior.
        // 2. We are unable to fetch block updates when we don't populate the
        //    blockstore with block history.
        // 3. We don't populate blockstore with block history when using the
        //    fast tx fail hotpath.
        // 4. Replay path always populates blockstore with block history.
        //
        // We trigger the entry point node to run the attacks, so by explicitly
        // not selecting that node here, we guarantee we will select a node that
        // will be replaying the malicious blocks.
        let addr = cluster
            .validators
            .iter()
            .find(|(_, v)| v.info.contact_info.pubkey() != cluster.entry_point_info.pubkey())
            .and_then(|(_, v)| v.info.contact_info.rpc_pubsub())
            .unwrap();

        PubsubClient::block_subscribe(
            format!("ws://{}", &addr.to_string()),
            RpcBlockSubscribeFilter::All,
            Some(RpcBlockSubscribeConfig {
                commitment: BLOCK_VALIDATION_COMMITMENT_LEVEL,
                encoding: None,
                transaction_details: Some(TransactionDetails::Signatures),
                show_rewards: None,
                max_supported_transaction_version: None,
            }),
        )
        .unwrap()
    }

    pub fn wait_for_first_epoch(client: &RpcClient) {
        loop {
            if client.get_epoch_info().unwrap().epoch > 0 {
                break;
            }
            sleep(Duration::from_secs(1));
        }
    }

    pub mod account {
        use super::*;

        pub fn simple_accounts(
            num_starting_accounts: usize,
            lamports_per_account: u64,
            space: usize,
            owner: &Pubkey,
        ) -> (Arc<Vec<Keypair>>, Vec<(Pubkey, AccountSharedData)>) {
            // Create a bunch of accounts to use as starting accounts for the generator.
            let keypairs: Arc<Vec<Keypair>> = Arc::new(
                iter::repeat_with(Keypair::new)
                    .take(num_starting_accounts)
                    .collect(),
            );
            let accounts: Vec<(Pubkey, AccountSharedData)> = keypairs
                .iter()
                .map(|k| {
                    (
                        k.pubkey(),
                        AccountSharedData::new(lamports_per_account, space, owner),
                    )
                })
                .collect();

            (keypairs, accounts)
        }

        /// Compiles block-generator-stress-test-program in the temporary directory and
        /// returns content of the generated so file.
        /// Specifically, it runs the following:
        /// cargo run --bin cargo-build-sbf --
        ///  --sbf-sdk "sdk/sbf"
        ///  --sbf-out-dir local-cluster/tests/program/
        ///  --manifest-path programs/block-generator-stress-test/Cargo.toml
        fn compile_test_program() -> Vec<u8> {
            // create a directory inside of std::env::temp_dir(), removed when goes out of scope
            let target_directory = tempdir().expect("temporary folder should be created");

            let manifest_directory = std::path::PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
            let source_directory = manifest_directory.join("..");

            let mut binding = std::process::Command::new(std::env!("CARGO"));
            let command = binding.current_dir(&source_directory).arg("run");

            if !cfg!(debug_assertions) {
                command.arg("--release");
            };

            command.args([
                "--bin",
                "cargo-build-sbf",
                "--",
                "--sbf-sdk",
                "sdk/sbf",
                "--sbf-out-dir",
                target_directory.path().to_str().unwrap(),
                "--manifest-path",
                "programs/block-generator-stress-test/Cargo.toml",
            ]);

            let output = command
                .output()
                .expect("block-generator-stress-test program should be successfully compiled");

            if !output.status.success() {
                let details = formatdoc! {"
                    Command: {:?}
                    Exit status: {:?}
                    std output:
                    {}
                    std error:
                    {}
                    ",
                    command,
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr),
                };
                panic!("block-generator-stress-test program compilation failed:\n{details}");
            }

            let target_file_name = "block_generator_stress_test.so";
            let target_path_name = target_directory.path().join(target_file_name);
            std::fs::read(target_path_name)
                .expect("Failed to read the program file.\nPath: {target_file_name}")
        }

        fn programs(num_programs: usize) -> (Arc<Vec<Pubkey>>, Vec<(Pubkey, AccountSharedData)>) {
            // Create a bunch of accounts to use as starting accounts for the generator.
            let pubkeys: Arc<Vec<Pubkey>> = Arc::new(if num_programs == 1 {
                // Special case. Assign static program ID so that it can be
                // used for verifying transactions later.
                vec![STRESS_TEST_PROGRAM_ID]
            } else {
                iter::repeat_with(Keypair::new)
                    .take(num_programs)
                    .map(|k| k.pubkey())
                    .collect()
            });
            // just use the same program repeatedly
            let program_data = if num_programs > 0 {
                // Only compile the program if we are going to use it. Compiling
                // it mucks with timestamps and can force long re-compiles
                // during development.
                compile_test_program()
            } else {
                vec![]
            };
            let accounts: Vec<(Pubkey, AccountSharedData)> = pubkeys
                .iter()
                .map(|k| {
                    (
                        *k,
                        AccountSharedData::from(Account {
                            lamports: Rent::default().minimum_balance(program_data.len()).min(1),
                            data: program_data.to_vec(),
                            owner: bpf_loader::id(),
                            executable: true,
                            rent_epoch: Epoch::MAX,
                        }),
                    )
                })
                .collect();

            (pubkeys, accounts)
        }

        fn get_account_num_of_each_type(attack: &Attack) -> (usize, usize, usize) {
            let num_replay_threads = 4;
            match attack {
                Attack::TransferRandom
                | Attack::CreateNonceAccounts
                | Attack::ChainTransactions
                | Attack::AllocateRandomSmall
                | Attack::AllocateRandomLarge
                | Attack::ReadNonExistentAccounts(_) => (1_000, 0, 0),
                Attack::ReadMaxSizeAccounts | Attack::WriteMaxSizeAccounts => {
                    let num_max_size_accounts = TX_MAX_NUM_MAX_SIZE_ACCOUNTS * BATCH_SIZE;
                    (BATCH_SIZE, num_max_size_accounts, 0)
                }
                Attack::WriteProgram(attack_config)
                | Attack::ReadProgram(attack_config)
                | Attack::RecursiveProgram(attack_config)
                | Attack::CpiProgram(attack_config) => {
                    let num_payers_accounts = attack_config
                        .transaction_batch_size
                        .saturating_mul(num_replay_threads);
                    let num_max_size_accounts = attack_config
                        .transaction_batch_size
                        .saturating_mul(attack_config.num_accounts_per_tx);
                    (num_payers_accounts, num_max_size_accounts, 1)
                }
                Attack::LargeNop(attack_config) => {
                    let num_payers_accounts = attack_config
                        .common
                        .transaction_batch_size
                        .saturating_mul(num_replay_threads)
                        // Larger number to avoid collisions because accounts
                        // get selected by random
                        .saturating_mul(10);
                    (num_payers_accounts, 0, 1)
                }
                Attack::ColdProgramCache(attack_config) => {
                    let num_payers_accounts = attack_config
                        .transaction_batch_size
                        .saturating_mul(num_replay_threads);
                    (num_payers_accounts, 0, num_payers_accounts)
                }
                _ => unimplemented!(),
            }
        }

        pub fn all_accounts(
            attack: &Attack,
            max_account_size: Option<usize>,
        ) -> (Arc<AccountsFile>, Vec<(Pubkey, AccountSharedData)>) {
            // 1. Determine num accounts of each type to create based on attack.
            let (num_payers_accounts, num_max_size_accounts, num_program_accounts) =
                get_account_num_of_each_type(attack);

            // 2. Create the accounts.
            let lamports_per_account = 1_000_000_000_000;
            let (program_pubkeys, program_accounts) = programs(num_program_accounts);
            let max_account_owner = *program_pubkeys.first().unwrap_or(&Pubkey::new_unique());
            let (payers_keypairs, payers_accounts) = simple_accounts(
                num_payers_accounts,
                lamports_per_account,
                0,
                &system_program::id(),
            );
            let (max_size_keypairs, max_size_accounts) =
                if let Some(max_account_size) = max_account_size {
                    simple_accounts(
                        num_max_size_accounts,
                        lamports_per_account,
                        max_account_size,
                        &max_account_owner,
                    )
                } else {
                    // No max size accounts need to be created.
                    (Arc::new(Vec::new()), Vec::new())
                };

            // 3. Concatenate accounts of each type into a single vector.
            let mut starting_accounts = Vec::new();
            starting_accounts.extend(payers_accounts.iter().cloned());
            starting_accounts.extend(max_size_accounts.iter().cloned());
            starting_accounts.extend(program_accounts.iter().cloned());

            // 4. Create struct to access accounts by type
            let accounts_file = Arc::new(AccountsFile::new(
                Some(max_account_owner),
                Some(&payers_keypairs),
                Some(&max_size_keypairs),
                Some(&program_pubkeys),
            ));

            (accounts_file, starting_accounts)
        }
    }
}

mod verify {
    use {
        super::*,
        agave_reserved_account_keys::ReservedAccountKeys,
        solana_message::SimpleAddressLoader,
        solana_transaction::{
            sanitized::{MessageHash, SanitizedTransaction},
            versioned::VersionedTransaction,
        },
        solana_transaction_status::{
            EncodedConfirmedTransactionWithStatusMeta, UiTransactionEncoding,
        },
    };

    // Large enough to ensure cluster is still making progress.
    const NUM_NEW_ROOTS_TO_WAIT_FOR: usize = 16;

    fn is_vote_transaction(tx: &VersionedTransaction) -> bool {
        SanitizedTransaction::try_create(
            tx.clone(),
            MessageHash::Compute,
            None,
            SimpleAddressLoader::Disabled,
            &ReservedAccountKeys::empty_key_set(),
        )
        .map(|tx| tx.is_simple_vote_transaction())
        .unwrap_or_default()
    }

    fn transaction_has_errors(transaction: &EncodedConfirmedTransactionWithStatusMeta) -> bool {
        transaction.transaction.meta.as_ref().unwrap().err.is_some()
    }

    // Queries RPC for the provided transaction signature and checks if it is
    // associated with a confirmed attack transaction.
    fn signature_is_for_attack_transaction(signature: &str, client: &RpcClient) -> bool {
        let config = solana_client::rpc_config::RpcTransactionConfig {
            encoding: Some(UiTransactionEncoding::Base64),
            commitment: BLOCK_VALIDATION_COMMITMENT_LEVEL,
            max_supported_transaction_version: Some(0),
        };

        match client.get_transaction_with_config(&signature.parse().unwrap(), config) {
            Ok(transaction) => {
                if transaction_has_errors(&transaction) {
                    panic!("Transaction has unexpected errors: {transaction:?}");
                }

                let decoded_tx = transaction
                    .transaction
                    .transaction
                    .decode()
                    .expect("Transaction must decode");
                if !is_vote_transaction(&decoded_tx) {
                    info!("Found attack transaction: {decoded_tx:?}");
                    return true;
                }
            }
            Err(err) => error!("Error getting transaction: {err:?}"),
        }

        false
    }

    fn block_includes_attack_txs(
        block: &Option<UiConfirmedBlock>,
        client: &RpcClient,
        use_failed_transaction_hotpath: bool,
    ) -> bool {
        // Extract the signatures from the block.
        let Some(UiConfirmedBlock {
            signatures: Some(signatures),
            ..
        }) = block
        else {
            // No signatures found.
            return false;
        };

        if use_failed_transaction_hotpath {
            // Failed tx hot path doesn't save tx history, so we can't verify
            // transaction content and status. Use transaction count as a proxy
            // for whether non-votes are making it into the block.
            if signatures.len() > NUM_NODES {
                return true;
            }
        } else {
            for signature in signatures {
                if signature_is_for_attack_transaction(signature, client) {
                    return true;
                }
            }
        }

        false
    }

    fn block_updates_include_attack_transactions(
        receiver: &Receiver<Response<RpcBlockUpdate>>,
        client: &RpcClient,
        use_failed_transaction_hotpath: bool,
    ) -> bool {
        // Each response is an update for a confirmed block.
        for response in receiver.try_iter() {
            let RpcBlockUpdate { block, err, .. } = response.value;
            assert_eq!(err, None);

            if block_includes_attack_txs(&block, client, use_failed_transaction_hotpath) {
                return true;
            }
        }

        false
    }

    pub fn attack_transactions_are_landing(
        receiver: &Receiver<Response<RpcBlockUpdate>>,
        client: &RpcClient,
        test_duration: Duration,
        num_response_check_iterations: u32,
        use_failed_transaction_hotpath: bool,
    ) {
        let check_sleep_duration = test_duration
            .checked_div(num_response_check_iterations)
            .expect("check iterations must be greater than zero");

        for _ in 0..num_response_check_iterations {
            if block_updates_include_attack_transactions(
                receiver,
                client,
                use_failed_transaction_hotpath,
            ) {
                return;
            }
            sleep(check_sleep_duration);
        }

        panic!("No attack transactions were found in the block updates");
    }

    // Verify the cluster is still making roots.
    fn cluster_advancing(cluster: &LocalCluster) {
        cluster.check_for_new_roots(
            NUM_NEW_ROOTS_TO_WAIT_FOR,
            "local_cluster_replay_attack",
            SocketAddrSpace::Unspecified,
        );
    }

    pub fn cluster_and_cleanup(
        cluster: LocalCluster,
        block_subscribe_client: &mut PubsubClientSubscription<Response<RpcBlockUpdate>>,
    ) {
        // This also provides time for attack transactions to drain out.
        cluster_advancing(&cluster);

        // If we don't drop the cluster, the blocking web socket service
        // won't return, and the `block_subscribe_client` won't shut down
        drop(cluster);
        block_subscribe_client.shutdown().unwrap();
    }

    pub fn program_was_deployed(client: &RpcClient, program_id: &Pubkey) {
        assert_ne!(
            client
                .get_account_with_commitment(program_id, CommitmentConfig::processed())
                .unwrap()
                .value,
            None
        );
    }

    // We expect at least one account to have been written to by the attack
    // because we previously verified that an attack transaction landed
    // successfully. This function confirms this by checking the first byte of
    // data that should have been updated by this transaction.
    pub fn at_least_one_account_written(
        client: &RpcClient,
        max_size_accounts_keypairs: &Vec<Keypair>,
        account_size: usize,
    ) {
        for account in max_size_accounts_keypairs {
            let account_data = client
                // Use processed because updated account data may not be confirmed yet.
                .get_account_with_commitment(&account.pubkey(), CommitmentConfig::processed())
                .expect("Account is present")
                .value
                .unwrap()
                .data;

            assert_eq!(account_data.len(), account_size);
            // Value for the first byte of the account is passed to the program in test_generator.
            if account_data[0] == 128u8 {
                return;
            }
        }

        // At least 1 account must have been modified if attack transactions
        // landed successfully.
        panic!("No accounts were found to be modified");
    }
}

fn use_failed_transaction_hotpath(attack: &Attack) -> bool {
    match attack {
        Attack::WriteProgram(attack_config)
        | Attack::ReadProgram(attack_config)
        | Attack::RecursiveProgram(attack_config)
        | Attack::CpiProgram(attack_config)
        | Attack::ColdProgramCache(attack_config) => attack_config.use_failed_transaction_hotpath,
        Attack::ReadNonExistentAccounts(attack_config) => {
            attack_config.use_failed_transaction_hotpath
        }
        _ => false,
    }
}

fn call_configure_replay_stage_attack(
    cluster: &LocalCluster,
    selected_attack: Option<Attack>,
) -> Result<(), String> {
    let url = &cluster_tests::get_rpc_url(cluster);
    let params = serde_json::json!([replay_stage_attack::AdversarialConfig { selected_attack }]);
    send_request_verified(url, "configureReplayStageAttack", params, None)?;
    Ok(())
}

fn start_replay_attack(cluster: &LocalCluster, attack: Attack) -> Result<(), String> {
    call_configure_replay_stage_attack(cluster, Some(attack))
}

fn stop_replay_attack(cluster: &LocalCluster) -> Result<(), String> {
    call_configure_replay_stage_attack(cluster, None)
}

fn run_replay_attack(attack: Attack) {
    solana_logger::setup_with_default(RUST_LOG_FILTER);

    // Setup the necessary accounts and programs.
    let max_account_size = match attack {
        Attack::WriteProgram(_)
        | Attack::ReadProgram(_)
        | Attack::RecursiveProgram(_)
        | Attack::CpiProgram(_) => Some(4 * 1024),
        Attack::ReadMaxSizeAccounts | Attack::WriteMaxSizeAccounts => Some(1),
        _ => None,
    };
    let (accounts_file, starting_accounts) =
        setup::account::all_accounts(&attack, max_account_size);

    // Setup cluster and clients.
    let cluster = setup::create_cluster(accounts_file.clone(), starting_accounts);
    let client = cluster
        .build_validator_tpu_quic_client(cluster.entry_point_info.pubkey())
        .unwrap();
    let rpc_client = client.rpc_client();

    // There can be partitioning issues when initially launching validators that
    // can result in some tx failures. Waiting for first epoch to complete
    // ensures nodes will be in sync and no transaction failures are expected.
    setup::wait_for_first_epoch(rpc_client);
    let (mut block_subscribe_client, receiver) = setup::block_subscriber(&cluster);

    // Check programs have been deployed.
    for program_id in &accounts_file.program_ids_jit_attack {
        verify::program_was_deployed(rpc_client, program_id);
    }

    match start_replay_attack(&cluster, attack.clone()) {
        Ok(_) => debug!("Replay attack {attack} started"),
        Err(err) => {
            panic!("Failed to start replay attack {attack}: {err}");
        }
    }

    // Let the cluster run for some period of time generating transactions.
    let max_test_duration = Duration::from_secs(15);
    let num_response_check_iterations = 5;

    // Verify transactions are landing during the attack.
    verify::attack_transactions_are_landing(
        &receiver,
        rpc_client,
        max_test_duration,
        num_response_check_iterations,
        use_failed_transaction_hotpath(&attack),
    );

    match stop_replay_attack(&cluster) {
        Ok(_) => debug!("Replay attack {attack} stopped"),
        Err(err) => {
            panic!("Failed to stop replay attack {attack}: {err}");
        }
    }

    // Perform post-attack verification and cleanup.
    if let Attack::WriteProgram(config) = attack {
        // State is not updated in the failed transaction hot path.
        if !config.use_failed_transaction_hotpath {
            verify::at_least_one_account_written(
                rpc_client,
                &accounts_file.max_size,
                max_account_size.expect("max_account_size must be Some for WriteProgram attack"),
            );
        }
    };
    verify::cluster_and_cleanup(cluster, &mut block_subscribe_client);
}

#[test]
#[serial]
fn test_transfer_random_generator() {
    run_replay_attack(Attack::TransferRandom);
}

#[test]
#[serial]
fn test_create_nonce_accounts_generator() {
    run_replay_attack(Attack::CreateNonceAccounts);
}

#[test]
#[serial]
fn test_chained_transactions_generator() {
    run_replay_attack(Attack::ChainTransactions);
}

#[test]
#[serial]
fn test_allocate_random_small_generator() {
    run_replay_attack(Attack::AllocateRandomSmall);
}

#[test]
#[serial]
fn test_allocate_random_large_generator() {
    run_replay_attack(Attack::AllocateRandomLarge);
}

#[test]
#[serial]
fn test_write_program_generator() {
    run_replay_attack(Attack::WriteProgram(AttackProgramConfig::default()));
}

#[test]
#[serial]
fn test_write_program_generator_fail_hotpath() {
    let config = AttackProgramConfig {
        transaction_cu_budget: 100,
        use_failed_transaction_hotpath: true,
        ..AttackProgramConfig::default()
    };
    run_replay_attack(Attack::WriteProgram(config));
}

#[test]
#[serial]
fn test_read_max_size_accounts_generator() {
    run_replay_attack(Attack::ReadMaxSizeAccounts);
}

#[test]
#[serial]
fn test_write_max_size_accounts_generator() {
    run_replay_attack(Attack::WriteMaxSizeAccounts);
}

#[test]
#[serial]
fn test_read_program_generator() {
    run_replay_attack(Attack::ReadProgram(AttackProgramConfig::default()));
}

#[test]
#[serial]
fn test_read_program_generator_fail_hotpath() {
    let config = AttackProgramConfig {
        transaction_cu_budget: 100,
        use_failed_transaction_hotpath: true,
        ..AttackProgramConfig::default()
    };
    run_replay_attack(Attack::ReadProgram(config));
}

#[test]
#[serial]
fn test_recursive_program_generator() {
    run_replay_attack(Attack::RecursiveProgram(AttackProgramConfig::default()));
}

#[test]
#[serial]
fn test_recursive_program_generator_fail_hotpath() {
    let config = AttackProgramConfig {
        transaction_cu_budget: 100,
        use_failed_transaction_hotpath: true,
        ..AttackProgramConfig::default()
    };
    run_replay_attack(Attack::RecursiveProgram(config));
}

#[test]
#[serial]
fn test_cpi_program_generator() {
    let config = AttackProgramConfig {
        // Currently each tx consumes a little over 1M CUs. This provides some
        // margin to allow tx to succeed.
        transaction_cu_budget: MAX_COMPUTE_UNIT_LIMIT,
        ..AttackProgramConfig::default()
    };
    run_replay_attack(Attack::CpiProgram(config));
}

#[test]
#[serial]
fn test_cpi_program_generator_fail_hotpath() {
    let config = AttackProgramConfig {
        transaction_cu_budget: 100,
        use_failed_transaction_hotpath: true,
        ..AttackProgramConfig::default()
    };
    run_replay_attack(Attack::CpiProgram(config));
}

#[test]
#[serial]
fn test_cold_program_cache_generator() {
    run_replay_attack(Attack::ColdProgramCache(AttackProgramConfig::default()));
}

#[test]
#[serial]
fn test_cold_program_cache_generator_fail_hotpath() {
    let config = AttackProgramConfig {
        transaction_cu_budget: 100,
        use_failed_transaction_hotpath: true,
        ..AttackProgramConfig::default()
    };
    run_replay_attack(Attack::ColdProgramCache(config));
}

#[test]
#[serial]
fn test_large_nop_generator() {
    run_replay_attack(Attack::LargeNop(LargeNopAttackConfig::default()));
}

#[test]
#[serial]
fn test_read_non_existent_accounts_generator() {
    run_replay_attack(Attack::ReadNonExistentAccounts(
        NonExistentAccountsAttackConfig::default(),
    ));
}

#[test]
#[serial]
#[ignore = "Test hangs. No attack tx detected. Dead blocks don't show up in the confirmed block \
            RPC updates."]
fn test_read_non_existent_accounts_generator_invalid_payer_hotpath() {
    let config = NonExistentAccountsAttackConfig {
        use_failed_transaction_hotpath: true,
        use_invalid_fee_payer: true,
    };
    run_replay_attack(Attack::ReadNonExistentAccounts(config));
}
