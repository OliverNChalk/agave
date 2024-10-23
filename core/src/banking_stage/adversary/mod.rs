pub mod adversarial_banking_stage;
pub mod args;
pub mod attack_scheduler;
pub mod bypass;
pub(crate) mod failed_transaction_hotpath;
mod generator_components;
pub mod generator_templates;
pub mod invalidate_leader_block_attack;
pub mod transaction_generators;

use {
    solana_adversary::adversary_feature_set::replay_stage_attack::Attack,
    std::{file, line},
};

/// Provides late binding for the attack configuration verification code in `adversary`.
///
/// `adversary` crate can not rely on the `core` crate, and thus, can not use compile time binding
/// to call different validation functions.
pub(crate) fn register_attack_config_verifiers() {
    macro_rules! verify_accounts {
        ($name:literal, $attack_module:ident) => {
            Attack::register_config_verifier(
                $name,
                concat!(file!(), ":", line!()),
                Box::new(move |accounts, _attack| {
                    transaction_generators::$attack_module::verify(accounts)
                }),
            )
            .expect("Successful registration");
        };
    }

    macro_rules! verify_accounts_and_attack_config {
        ($name:literal, $attack_module:ident) => {
            Attack::register_config_verifier(
                $name,
                concat!(file!(), ":", line!()),
                Box::new(move |accounts, attack| {
                    transaction_generators::$attack_module::verify(accounts, attack)
                }),
            )
            .expect("Successful registration");
        };
    }

    verify_accounts!("transferRandom", transfer_random);
    verify_accounts!("transferRandomWithMemo", transfer_random_with_memo);
    verify_accounts!("createNonceAccounts", create_nonce_accounts);
    verify_accounts!("allocateRandomLarge", allocate_random_large);
    verify_accounts!("allocateRandomSmall", allocate_random_small);
    verify_accounts!("chainTransactions", chain_transactions);
    verify_accounts_and_attack_config!("writeProgram", write_program);
    verify_accounts!("readMaxSizeAccounts", read_max_size_accounts);
    verify_accounts!("writeMaxSizeAccounts", write_max_size_accounts);
    verify_accounts_and_attack_config!("readProgram", read_program);
    verify_accounts_and_attack_config!("recursiveProgram", recursive_program);
    verify_accounts_and_attack_config!("cpiProgram", cpi_program);
    verify_accounts_and_attack_config!("coldProgramCache", cold_program_cache);
    verify_accounts_and_attack_config!("largeNop", large_nop);
    verify_accounts_and_attack_config!("readNonExistentAccounts", read_non_existent_accounts);

    Attack::end_verifier_registration().expect("All config verifiers are registered");
}

#[cfg(test)]
mod test_helpers {
    use {
        super::register_attack_config_verifiers,
        solana_adversary::{
            accounts_file::AccountsFile,
            adversary_feature_set::replay_stage_attack::{set_config, AdversarialConfig},
            auth::JsonRpcAuthToken,
            block_generator_config::{BlockGeneratorAccountsSource, BlockGeneratorConfig},
        },
        solana_keypair::Keypair,
        solana_ledger::genesis_utils::GenesisConfigInfo,
        solana_rpc::{
            rpc::JsonRpcRequestProcessor,
            rpc_adversary::{test_helpers::setup_test as setup_test_from_rpc, MetaIoWithAuth},
        },
        solana_runtime::{bank::Bank, genesis_utils::create_genesis_config},
        solana_signer::Signer,
        std::{iter, sync::Arc},
    };

    /// Should be used instead of [`solana_rpc::rpc_adversary::setup_test()`], as resets the global
    /// replay attack configuration state to the default at the beginning of each test.  And runs
    /// [`register_attack_config_verifiers()`], to allow correct RPC endpoint verification in tests.
    ///
    /// This only applies to tests that configure replay stage attacks.
    pub(super) fn setup_test() -> (
        JsonRpcRequestProcessor,
        Keypair,
        MetaIoWithAuth,
        JsonRpcAuthToken,
    ) {
        // As configuration is a singleton it is preserved from one test execution to another.
        // So we clean it every time we setup test environment.
        // We should switch to a more composable design for the configuration.
        set_config(AdversarialConfig::default());

        register_attack_config_verifiers();

        setup_test_from_rpc()
    }

    pub(super) fn create_test_bank() -> Arc<Bank> {
        let GenesisConfigInfo { genesis_config, .. } = create_genesis_config(10_000);
        Bank::new_no_wallclock_throttle_for_tests(&genesis_config).0
    }

    pub(super) struct TestAccounts {
        program_id: Keypair,
        payers: Vec<Keypair>,
        max_size: Vec<Keypair>,
    }

    impl TestAccounts {
        pub(super) fn new(payers_count: usize, max_size_count: usize) -> Self {
            let program_id = Keypair::new();
            let payers = iter::repeat_with(Keypair::new)
                .take(payers_count)
                .collect::<Vec<_>>();
            let max_size = iter::repeat_with(Keypair::new)
                .take(max_size_count)
                .collect::<Vec<_>>();

            TestAccounts {
                program_id,
                payers,
                max_size,
            }
        }
    }

    impl From<TestAccounts> for AccountsFile {
        fn from(
            TestAccounts {
                program_id,
                payers,
                max_size,
            }: TestAccounts,
        ) -> Self {
            Self {
                owner_program_id: Some(program_id.pubkey()),
                payers,
                max_size,
                ..Default::default()
            }
        }
    }

    /// Populates the [`JsonRpcRequestProcessor::block_generator_config`] field.
    pub(super) fn setup_accounts(
        meta: &mut JsonRpcRequestProcessor,
        payers_count: usize,
        max_size_count: usize,
    ) {
        let TestAccounts {
            program_id,
            payers,
            max_size,
        } = TestAccounts::new(payers_count, max_size_count);

        let accounts = BlockGeneratorAccountsSource::Genesis(Arc::new(AccountsFile::new(
            Some(program_id.pubkey()),
            Some(&payers),
            Some(&max_size),
            None,
        )));

        *(meta.block_generator_config_mut()) = Some(BlockGeneratorConfig { accounts });
    }
}
