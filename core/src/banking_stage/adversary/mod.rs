pub mod adversarial_banking_stage;
pub mod args;
pub mod attack_scheduler;
pub mod bypass;
pub(crate) mod failed_transaction_hotpath;
pub mod invalidate_leader_block_attack;
pub mod transaction_generators;

#[cfg(test)]
mod test_helpers {
    use {
        solana_adversary::{
            accounts_file::AccountsFile,
            adversary_feature_set::replay_stage_attack::{set_config, AdversarialConfig},
            auth::JsonRpcAuthToken,
            block_generator_config::{BlockGeneratorAccountsSource, BlockGeneratorConfig},
        },
        solana_keypair::Keypair,
        solana_rpc::{
            rpc::JsonRpcRequestProcessor,
            rpc_adversary::{test_helpers::setup_test as setup_test_from_rpc, MetaIoWithAuth},
        },
        solana_signer::Signer,
        std::{iter, sync::Arc},
    };

    /// Should be used instead of [`solana_rpc::rpc_adversary::setup_test()`], as resets the global
    /// replay attack configuration state to the default at the beginning of each test.
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

        setup_test_from_rpc()
    }

    /// Populates the [`JsonRpcRequestProcessor::block_generator_config`] field.
    pub(super) fn setup_accounts(
        meta: &mut JsonRpcRequestProcessor,
        payers_count: usize,
        max_size_count: usize,
    ) {
        let program_id = Keypair::new();
        let payers = Arc::new(
            iter::repeat_with(Keypair::new)
                .take(payers_count)
                .collect::<Vec<_>>(),
        );
        let max_size = Arc::new(
            iter::repeat_with(Keypair::new)
                .take(max_size_count)
                .collect::<Vec<_>>(),
        );
        *(meta.block_generator_config_mut()) = Some(BlockGeneratorConfig {
            accounts: BlockGeneratorAccountsSource::Genesis(Arc::new(
                AccountsFile::with_payers_and_max_size(&program_id.pubkey(), &payers, &max_size),
            )),
        });
    }
}
