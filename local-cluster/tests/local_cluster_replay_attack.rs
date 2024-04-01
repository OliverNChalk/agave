use {
    agave_reserved_account_keys::ReservedAccountKeys,
    crossbeam_channel::Receiver,
    log::debug,
    serial_test::serial,
    solana_account::{Account, AccountSharedData},
    solana_adversary::{
        accounts_file::AccountsFile,
        adversary_feature_set::replay_stage_attack::{
            self, Attack, AttackProgramConfig, LargeNopAttackConfig,
        },
        block_generator_config::{BlockGeneratorAccountsSource, BlockGeneratorConfig},
        send_request_verified,
    },
    solana_bincode::limited_deserialize,
    solana_client::{pubsub_client::PubsubClientSubscription, rpc_response::RpcBlockUpdate},
    solana_cluster_type::ClusterType,
    solana_commitment_config::CommitmentConfig,
    solana_core::validator::{InvalidatorConfig, ValidatorConfig},
    solana_gossip::gossip_service::discover_validators,
    solana_keypair::Keypair,
    solana_local_cluster::{
        cluster::Cluster,
        cluster_tests,
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
        response::{Response, RpcBlockUpdateError},
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

const STRESS_TEST_PROGRAM_ID: Pubkey = Pubkey::new_from_array([7u8; 32]);

trait TestVersionedTransaction {
    fn is_simple_vote(&self) -> bool;
    fn is_transfer(&self) -> bool;
    fn is_allocate(&self) -> bool;
    fn is_calling_stress_test_program(&self) -> bool;
    fn is_calling_user_program(&self) -> bool;
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

    // Return true if transaction contains single Allocate instruction
    fn is_allocate(&self) -> bool {
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
                .map(|instruction| matches!(instruction, SystemInstruction::Allocate { space: _ }))
                .unwrap_or_default();
        }
        false
    }

    // Return true if transaction is calling the stress test program
    fn is_calling_stress_test_program(&self) -> bool {
        let message = &self.message;
        let account_keys = message.static_account_keys();

        let program_ids: Vec<_> = message
            .instructions()
            .iter()
            .map(|ix| ix.program_id(account_keys))
            .collect();
        program_ids.len() == 2 && *program_ids[1] == STRESS_TEST_PROGRAM_ID
    }

    // Return true if transaction is calling a user program that is not the
    // stress test program.
    fn is_calling_user_program(&self) -> bool {
        let message = &self.message;
        let account_keys = message.static_account_keys();

        let program_ids: Vec<_> = message
            .instructions()
            .iter()
            .map(|ix| ix.program_id(account_keys))
            .collect();
        if program_ids.len() == 2 {
            match *program_ids[1] {
                STRESS_TEST_PROGRAM_ID => false,
                _ if *program_ids[1] == system_program::id() => false,
                _ => true,
            }
        } else {
            false
        }
    }
}
mod setup {
    use super::*;

    pub fn create_cluster(
        num_nodes: usize,
        accounts_file: Arc<AccountsFile>,
        starting_accounts: Vec<(Pubkey, AccountSharedData)>,
    ) -> LocalCluster {
        // Create a validator config configured for block generation.
        let mut validator_config = ValidatorConfig {
            invalidator_config: InvalidatorConfig {
                block_generator_config: Some(BlockGeneratorConfig {
                    accounts: BlockGeneratorAccountsSource::Genesis(accounts_file),
                }),
                rpc_adversary_id: None,
            },
            ..ValidatorConfig::default_for_test()
        };
        validator_config.enable_default_rpc_block_subscribe();

        // Create a cluster config with the generator validator config.
        let mut config = ClusterConfig {
            cluster_type: ClusterType::Development,
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
            cluster.entry_point_info.shred_version(),
            SocketAddrSpace::Unspecified,
        )
        .unwrap();
        assert_eq!(cluster_nodes.len(), num_nodes);

        cluster
    }

    pub fn block_subscriber(
        cluster: &LocalCluster,
    ) -> (
        PubsubClientSubscription<Response<RpcBlockUpdate>>,
        Receiver<Response<RpcBlockUpdate>>,
    ) {
        PubsubClient::block_subscribe(
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
        .unwrap()
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

            let _output = command
                .output()
                .expect("block-generator-stress-test program should be successfully compiled");

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
            let program_data = compile_test_program();
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
                | Attack::ChainTransactions
                | Attack::AllocateRandomSmall
                | Attack::AllocateRandomLarge => (1_000, 0, 0),
                Attack::WriteProgram(attack_config) | Attack::ReadProgram(attack_config) => {
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
                        .saturating_mul(num_replay_threads);
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

            // 4. Create an accounts file based on the attack.
            let accounts_file = match attack {
                Attack::TransferRandom
                | Attack::ChainTransactions
                | Attack::AllocateRandomSmall
                | Attack::AllocateRandomLarge => {
                    Arc::new(AccountsFile::with_payers(&payers_keypairs))
                }
                Attack::WriteProgram(_) | Attack::ReadProgram(_) | Attack::LargeNop(_) => {
                    Arc::new(AccountsFile::with_payers_and_max_size(
                        &max_account_owner,
                        &payers_keypairs,
                        &max_size_keypairs,
                    ))
                }
                Attack::ColdProgramCache(_) => Arc::new(AccountsFile::with_payers_and_programs(
                    &payers_keypairs,
                    &program_pubkeys,
                )),
                _ => unimplemented!(),
            };

            (accounts_file, starting_accounts)
        }
    }
}

mod verify {
    use {super::*, solana_client::rpc_client::RpcClient};

    // This number must be large enough to allow ample time for attack
    // transactions to drain out.
    const NUM_NEW_ROOTS_TO_WAIT_FOR: usize = 16;

    pub fn attack_transactions_are_landing(
        receiver: &Receiver<Response<RpcBlockUpdate>>,
        test_duration: Duration,
        num_response_check_iterations: u32,
        valid_transaction_check: impl Fn(&VersionedTransaction) -> bool,
    ) {
        let check_sleep_duration = test_duration
            .checked_div(num_response_check_iterations)
            .expect("check iterations must be greater than zero");
        let mut num_valid_txs: i32 = 0;
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
                                if valid_transaction_check(&tx) {
                                    num_valid_txs = num_valid_txs.saturating_add(1);
                                }
                            }
                        }
                    }
                }
            });
            sleep(check_sleep_duration);
        }
        debug!("total valid transactions landed: {num_valid_txs}");
        assert_ne!(num_valid_txs, 0);
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
        receiver: Receiver<Response<RpcBlockUpdate>>,
        cluster: LocalCluster,
        block_subscribe_client: &mut PubsubClientSubscription<Response<RpcBlockUpdate>>,
    ) {
        // This also provides time for attack transactions to drain out.
        cluster_advancing(&cluster);

        // Clean the receiver.
        receiver.try_iter().for_each(drop);

        // Wait a bit so there are new block updates.
        sleep(Duration::from_secs(1));

        // Verify that there are no new attack transactions.
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
                            // Only expect to see votes because we allowed other
                            // transactions to drain out while checking the
                            // cluster was advancing.
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

    pub fn program_was_deployed(client: &RpcClient, program_id: &Pubkey) {
        assert_ne!(
            client
                .get_account_with_commitment(program_id, CommitmentConfig::processed())
                .unwrap()
                .value,
            None
        );
    }

    pub fn accounts_were_written(
        client: &RpcClient,
        max_size_accounts_keypairs: &Vec<Keypair>,
        account_size: usize,
    ) {
        for account in max_size_accounts_keypairs {
            let account_data = client
                .get_account_data(&account.pubkey())
                .expect("Account is present");

            assert_eq!(account_data.len(), account_size);
            // Value for the first byte of the account is passed to the program in test_generator.
            assert_eq!(account_data[0], 128u8);
        }
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
        Attack::WriteProgram(_) | Attack::ReadProgram(_) => Some(4 * 1024),
        Attack::TransferRandom
        | Attack::ChainTransactions
        | Attack::AllocateRandomLarge
        | Attack::AllocateRandomSmall
        | Attack::ColdProgramCache(_)
        | Attack::LargeNop(_) => None,
        _ => unimplemented!(),
    };
    let (accounts_file, starting_accounts) =
        setup::account::all_accounts(&attack, max_account_size);

    // Setup cluster and clients.
    let num_nodes = 2;
    let cluster = setup::create_cluster(num_nodes, accounts_file.clone(), starting_accounts);
    let client = cluster
        .build_validator_tpu_quic_client(cluster.entry_point_info.pubkey())
        .unwrap();
    let rpc_client = client.rpc_client();
    let (mut block_subscribe_client, receiver) = setup::block_subscriber(&cluster);

    // Check programs have been deployed.
    for program_id in &accounts_file.program_ids_jit_attack {
        verify::program_was_deployed(rpc_client, program_id);
    }
    sleep(Duration::from_millis(800));

    assert_eq!(start_replay_attack(&cluster, attack.clone()), Ok(()));

    // Let the cluster run for some period of time generating transactions.
    let test_duration = Duration::from_secs(15);
    let num_response_check_iterations = 5;
    let valid_transaction_check = match attack {
        Attack::TransferRandom | Attack::ChainTransactions => VersionedTransaction::is_transfer,
        Attack::AllocateRandomLarge | Attack::AllocateRandomSmall => {
            VersionedTransaction::is_allocate
        }
        Attack::WriteProgram(_) | Attack::ReadProgram(_) | Attack::LargeNop(_) => {
            VersionedTransaction::is_calling_stress_test_program
        }
        Attack::ColdProgramCache(_) => VersionedTransaction::is_calling_user_program,
        _ => unimplemented!(),
    };

    // Verify transactions are landing during the attack.
    verify::attack_transactions_are_landing(
        &receiver,
        test_duration,
        num_response_check_iterations,
        valid_transaction_check,
    );

    assert_eq!(stop_replay_attack(&cluster), Ok(()));

    // Perform post-attack verification and cleanup.
    if let Attack::WriteProgram(_) = attack {
        verify::accounts_were_written(
            rpc_client,
            &accounts_file.max_size,
            max_account_size.expect("max_account_size must be Some for WriteProgram attack"),
        );
    };
    verify::cluster_and_cleanup(receiver, cluster, &mut block_subscribe_client);
}

#[test]
#[serial]
fn test_transfer_random_generator() {
    run_replay_attack(Attack::TransferRandom);
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
fn test_read_program_generator() {
    run_replay_attack(Attack::ReadProgram(AttackProgramConfig::default()));
}

#[test]
#[serial]
fn test_cold_program_cache_generator() {
    run_replay_attack(Attack::ColdProgramCache(AttackProgramConfig::default()));
}

#[test]
#[serial]
fn test_large_nop_generator() {
    run_replay_attack(Attack::LargeNop(LargeNopAttackConfig::default()));
}
