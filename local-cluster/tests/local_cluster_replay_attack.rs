use {
    agave_reserved_account_keys::ReservedAccountKeys,
    serial_test::serial,
    solana_account::{Account, AccountSharedData},
    solana_adversary::{
        accounts_file::AccountsFile,
        adversary_feature_set::replay_stage_attack,
        block_generator_config::{BlockGeneratorAccountsSource, BlockGeneratorConfig},
        send_request_verified,
    },
    solana_bincode::limited_deserialize,
    solana_cluster_type::ClusterType,
    solana_commitment_config::CommitmentConfig,
    solana_core::validator::{InvalidatorConfig, ValidatorConfig},
    solana_gossip::gossip_service::discover_validators,
    solana_keypair::Keypair,
    solana_local_cluster::{
        cluster::Cluster as _,
        integration_tests::{DEFAULT_NODE_STAKE, RUST_LOG_FILTER},
        local_cluster::{ClusterConfig, LocalCluster, DEFAULT_MINT_LAMPORTS},
        validator_configs::*,
    },
    solana_message::SimpleAddressLoader,
    solana_pubkey::Pubkey,
    solana_pubsub_client::pubsub_client::PubsubClient,
    solana_rent::Rent,
    solana_rpc_client_api::{
        config::{RpcBlockSubscribeConfig, RpcBlockSubscribeFilter},
        response::RpcBlockUpdateError,
    },
    solana_sdk_ids::{bpf_loader, system_program},
    solana_signer::Signer,
    solana_stake_interface::stake_history::Epoch,
    solana_streamer::socket::SocketAddrSpace,
    solana_system_interface::instruction::SystemInstruction,
    solana_transaction::{
        sanitized::{MessageHash, SanitizedTransaction},
        versioned::VersionedTransaction,
    },
    std::{iter, sync::Arc, thread::sleep, time::Duration},
    tempfile::tempdir,
};

// Adversary tests are using global state config
// This struct is used to reset the state of AdversarialConfig
struct TestSetup;

impl Drop for TestSetup {
    fn drop(&mut self) {
        solana_adversary::adversary_feature_set::replay_stage_attack::set_config(
            replay_stage_attack::AdversarialConfig {
                selected_attack: None,
            },
        );
    }
}

trait TestVersionedTransaction {
    fn is_simple_vote(&self) -> bool;
    fn is_transfer(&self) -> bool;
}

impl TestVersionedTransaction for VersionedTransaction {
    fn is_simple_vote(&self) -> bool {
        // although this is, probably, not the most efficient way to check vote transaction
        // it is better to stick with SanitizedTransaction implementation to avoid inconsistent implementation
        SanitizedTransaction::try_create(
            self.clone(),
            MessageHash::Compute,
            None,
            SimpleAddressLoader::Disabled,
            &ReservedAccountKeys::empty_key_set(),
        )
        .map(|tx| tx.is_simple_vote_transaction())
        .unwrap_or_default()
    }

    // Return true if transaction contains single Transfer instruction
    fn is_transfer(&self) -> bool {
        let message = &self.message;
        let account_keys = message.static_account_keys();

        let program_ids: Vec<_> = message
            .instructions()
            .iter()
            .map(|ix| ix.program_id(account_keys))
            .collect();

        if program_ids.len() == 1 && system_program::check_id(program_ids[0]) {
            let data = &message.instructions()[0].data;
            return limited_deserialize(data, solana_packet::PACKET_DATA_SIZE as u64)
                .map(|instruction| {
                    matches!(instruction, SystemInstruction::Transfer { lamports: _ })
                })
                .unwrap_or_default();
        }
        false
    }
}

fn call_configure_replay_stage_attack(
    selected_attack: Option<replay_stage_attack::Attack>,
    url: &str,
) -> Result<(), String> {
    let params = serde_json::json!([replay_stage_attack::AdversarialConfig { selected_attack }]);
    send_request_verified(url, "configureReplayStageAttack", params, None)?;
    Ok(())
}

fn get_rpc_url(cluster: &LocalCluster) -> String {
    let rpc = cluster.entry_point_info.rpc().unwrap();
    format!("http://{}:{}", rpc.ip(), rpc.port())
}

#[test]
#[serial]
fn test_mainnet_beta_cluster_type_generator() {
    solana_logger::setup_with_default(RUST_LOG_FILTER);
    let _ = TestSetup;

    let num_nodes = 2;
    let test_duration = Duration::from_secs(20);
    let num_starting_accounts = 10_000;
    let lamports_per_account = 1_000_000_000_000;

    // Create a bunch of accounts to use as starting accounts for the generator.
    let starting_keypairs: Arc<Vec<Keypair>> = Arc::new(
        iter::repeat_with(Keypair::new)
            .take(num_starting_accounts)
            .collect(),
    );
    let starting_accounts: Vec<(Pubkey, AccountSharedData)> = starting_keypairs
        .iter()
        .map(|k| {
            (
                k.pubkey(),
                AccountSharedData::new(lamports_per_account, 0, &system_program::id()),
            )
        })
        .collect();

    // Create a validator config configured for block generation.
    let mut validator_config = ValidatorConfig {
        invalidator_config: InvalidatorConfig {
            block_generator_config: Some(BlockGeneratorConfig {
                accounts: BlockGeneratorAccountsSource::Genesis(Arc::new(
                    AccountsFile::with_payers(&starting_keypairs),
                )),
            }),
            rpc_adversary_id: None,
        },
        ..ValidatorConfig::default_for_test()
    };
    validator_config.enable_default_rpc_block_subscribe();

    // Create a cluster config with the generator validator config.
    let mut config = ClusterConfig {
        cluster_type: ClusterType::MainnetBeta,
        node_stakes: vec![DEFAULT_NODE_STAKE; num_nodes],
        mint_lamports: DEFAULT_MINT_LAMPORTS,
        validator_configs: make_identical_validator_configs(&validator_config, num_nodes),
        additional_accounts: starting_accounts,
        ..ClusterConfig::default()
    };

    // Create the cluster.
    let cluster = LocalCluster::new(&mut config, SocketAddrSpace::Unspecified);
    let cluster_nodes = discover_validators(
        &cluster.entry_point_info.gossip().unwrap(),
        num_nodes,
        0,
        SocketAddrSpace::Unspecified,
    )
    .unwrap();
    assert_eq!(cluster_nodes.len(), num_nodes);

    let (mut block_subscribe_client, receiver) = PubsubClient::block_subscribe(
        format!(
            "ws://{}",
            &cluster.entry_point_info.rpc_pubsub().unwrap().to_string()
        ),
        RpcBlockSubscribeFilter::All,
        Some(RpcBlockSubscribeConfig {
            commitment: Some(CommitmentConfig::confirmed()),
            encoding: None,
            transaction_details: None,
            show_rewards: None,
            max_supported_transaction_version: None,
        }),
    )
    .unwrap();

    sleep(Duration::from_millis(800));
    assert!(call_configure_replay_stage_attack(
        Some(replay_stage_attack::Attack::TransferRandom),
        &get_rpc_url(&cluster)
    )
    .is_ok());

    // check that the leader generated transactions that call transfer system instruction
    let num_response_check_iterations = 5;
    let check_sleep_duration = test_duration / num_response_check_iterations;
    let mut num_transfer_txs = 0;
    for _ in 0..num_response_check_iterations {
        receiver.try_iter().for_each(|response| {
            if let Some(err) = response.value.err {
                // sometimes block is not ready, see issues/33462
                assert_eq!(err, RpcBlockUpdateError::BlockStoreError);
            }
            if let Some(block) = response.value.block {
                if let Some(encoded_transactions) = block.transactions {
                    for encoded_tx in encoded_transactions {
                        let tx = encoded_tx.transaction.decode();
                        if let Some(tx) = tx {
                            if tx.is_transfer() {
                                num_transfer_txs += 1;
                            }
                        }
                    }
                }
            }
        });
        sleep(check_sleep_duration);
    }
    assert_ne!(num_transfer_txs, 0);
    assert_eq!(
        call_configure_replay_stage_attack(None, &get_rpc_url(&cluster)),
        Ok(())
    );
    // Check that the cluster is making progress
    cluster.check_for_new_roots(
        16,
        "test_mainnet_beta_cluster_type_generator",
        SocketAddrSpace::Unspecified,
    );

    // clean the receiver
    receiver.try_iter().for_each(drop);
    // verify that there are no new transfer transactions
    // wait for a while to have some vote transactions
    sleep(Duration::from_secs(1));
    receiver.try_iter().for_each(|response| {
        if let Some(err) = response.value.err {
            // sometimes block is not ready, see issues/33462
            assert_eq!(err, RpcBlockUpdateError::BlockStoreError);
        }
        if let Some(block) = response.value.block {
            if let Some(encoded_transactions) = block.transactions {
                for encoded_tx in encoded_transactions {
                    let tx = encoded_tx.transaction.decode();
                    if let Some(tx) = tx {
                        assert!(tx.is_simple_vote());
                    }
                }
            }
        }
    });

    // If we don't drop the cluster, the blocking web socket service
    // won't return, and the `block_subscribe_client` won't shut down
    drop(cluster);
    block_subscribe_client.shutdown().unwrap();
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

    let _output = command
        .output()
        .expect("block-generator-stress-test program should be successfully compiled");

    let target_file_name = "block_generator_stress_test.so";
    let target_path_name = target_directory.path().join(target_file_name);
    std::fs::read(target_path_name)
        .expect("Failed to read the program file.\nPath: {target_file_name}")
}

fn program_account() -> AccountSharedData {
    let program_data = compile_test_program();
    AccountSharedData::from(Account {
        lamports: Rent::default().minimum_balance(program_data.len()).min(1),
        data: program_data.to_vec(),
        owner: bpf_loader::id(),
        executable: true,
        rent_epoch: Epoch::MAX,
    })
}

#[test]
#[serial]
fn test_mainnet_beta_cluster_type_program_generator() {
    solana_logger::setup_with_default(RUST_LOG_FILTER);
    let _ = TestSetup;

    let attack_config = replay_stage_attack::AttackProgramConfig::default();
    // For the sake of this test, check with smaller than
    // `solana_system_interface::MAX_PERMITTED_DATA_LENGTH` accounts to avoid too large
    // archive problem due to ledger genesis archive size limit (see
    // MAX_GENESIS_ARCHIVE_UNPACKED_SIZE).
    let account_size = 4 * 1024;

    let num_nodes = 2;
    let test_duration = Duration::from_secs(30);
    let num_max_size_accounts =
        attack_config.transaction_batch_size * attack_config.num_accounts_per_tx;
    // To avoid AccountInUse, each transaction has it's own payer
    let num_payers_accounts = attack_config.transaction_batch_size;
    let num_program_accounts = 1;
    let lamports_per_account = 1_000_000_000_000;

    // Create a bunch of accounts to use as starting accounts for the generator.
    // the layout of this array is the following:
    // [payers_start_index..max_account_start_index] -- payers accounts,
    // [max_account_start_index..program_index] -- large accounts
    // [program_index] -- program account
    let payers_start_index = 0usize;
    let max_account_start_index = num_payers_accounts;
    let program_index = max_account_start_index + num_max_size_accounts;
    let starting_keypairs: Arc<Vec<Keypair>> = Arc::new(
        iter::repeat_with(Keypair::new)
            .take(num_payers_accounts + num_max_size_accounts + num_program_accounts)
            .collect(),
    );

    // add program account
    let stress_test_program_keypair = &starting_keypairs[program_index];
    let stress_test_program_id = stress_test_program_keypair.pubkey();

    let payers_accounts = &starting_keypairs[payers_start_index..max_account_start_index];
    let max_size_accounts = &starting_keypairs[max_account_start_index..program_index];
    let accounts_file = Arc::new(AccountsFile::with_payers_and_max_size(
        &stress_test_program_id,
        payers_accounts,
        max_size_accounts,
    ));

    // Create a validator config with a generator account config.
    let mut validator_config = ValidatorConfig::default_for_test();
    validator_config.invalidator_config.rpc_adversary_id = None;
    validator_config.invalidator_config.block_generator_config = Some(BlockGeneratorConfig {
        accounts: BlockGeneratorAccountsSource::Genesis(accounts_file),
    });

    // Setup starting accounts
    let mut starting_accounts: Vec<(Pubkey, AccountSharedData)> =
        Vec::with_capacity(starting_keypairs.len());
    for account_keypair in payers_accounts {
        starting_accounts.push((
            account_keypair.pubkey(),
            AccountSharedData::new(lamports_per_account, 0, &system_program::id()),
        ));
    }
    for account_keypair in max_size_accounts {
        starting_accounts.push((
            account_keypair.pubkey(),
            AccountSharedData::new(lamports_per_account, account_size, &stress_test_program_id),
        ));
    }

    starting_accounts.push((stress_test_program_id, program_account()));

    // Create a cluster config with the generator validator config.
    let mut config = ClusterConfig {
        cluster_type: ClusterType::MainnetBeta,
        node_stakes: vec![DEFAULT_NODE_STAKE; num_nodes],
        mint_lamports: DEFAULT_MINT_LAMPORTS,
        validator_configs: make_identical_validator_configs(&validator_config, num_nodes),
        additional_accounts: starting_accounts,
        ..ClusterConfig::default()
    };

    // Create the cluster.
    let cluster = LocalCluster::new(&mut config, SocketAddrSpace::Unspecified);
    let cluster_nodes = discover_validators(
        &cluster.entry_point_info.gossip().unwrap(),
        num_nodes,
        0,
        SocketAddrSpace::Unspecified,
    )
    .unwrap();
    assert_eq!(cluster_nodes.len(), num_nodes);

    let tpu_client = cluster
        .build_validator_tpu_quic_client(cluster.entry_point_info.pubkey())
        .unwrap();
    let rpc_client = tpu_client.rpc_client();

    // Make sure the program account is part of the blockchain.
    assert_ne!(
        rpc_client
            .get_account_with_commitment(&stress_test_program_id, CommitmentConfig::processed())
            .unwrap()
            .value,
        None
    );

    sleep(Duration::from_millis(800));
    assert_eq!(
        call_configure_replay_stage_attack(
            Some(replay_stage_attack::Attack::WriteProgram(attack_config)),
            &get_rpc_url(&cluster)
        ),
        Ok(())
    );

    // Let the cluster run for some period of time generating transactions.
    std::thread::sleep(test_duration);

    // Check that accounts has been modified which means that programs have been executed at least
    // once.
    for account in max_size_accounts {
        let account_data = rpc_client
            .get_account_data(&account.pubkey())
            .expect("Account is present");

        assert_eq!(account_data.len(), account_size);
        // Value for the first byte of the account is passed to the program in test_generator.
        assert_eq!(account_data[0], 128u8);
    }
}
