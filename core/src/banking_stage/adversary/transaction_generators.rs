//! Generators for testing banking stage

use {
    solana_adversary::{
        accounts_file::AccountsFile, adversary_feature_set::replay_stage_attack::Attack,
        SelectedReplayAttack,
    },
    solana_runtime::bank::Bank,
    solana_transaction::sanitized::SanitizedTransaction,
    std::sync::Arc,
};

pub(super) mod allocate_random_large;
pub(super) mod allocate_random_small;
pub(super) mod chain_transactions;
pub(super) mod cold_program_cache;
pub(super) mod create_nonce_accounts;
pub(super) mod large_nop;
pub(super) mod read_max_accounts;
pub(super) mod read_non_existent_accounts;
pub(super) mod read_program;
pub(super) mod recursive_program;
pub(super) mod transfer_random;
pub(super) mod transfer_random_with_memo;
pub(super) mod write_max_accounts;
pub(super) mod write_program;

/// Attacks are essentially sequences of transactions.  Each will have a function that is
/// responsible for generating these transactions, and those functions will adhere to this type.
pub type TransactionGenerator = Box<dyn Send + FnMut(&Bank) -> (Vec<SanitizedTransaction>, usize)>;

/// Encapsulate logic for managing selected generator
pub struct ActiveGenerator {
    attack: Attack,
    generator: TransactionGenerator,
    execution_tx_batch_size: usize,
}

impl ActiveGenerator {
    /// Produces an `ActiveGenerator` if the `SelectedReplayAttack` indicates an attack needs to be
    /// run.
    pub fn given(selected_attack: SelectedReplayAttack, num_workers: usize) -> Option<Self> {
        let (accounts, attack) = match selected_attack {
            SelectedReplayAttack::Selected { accounts, attack } => (accounts, attack),
            SelectedReplayAttack::None => return None,
        };

        info!("Reset selected generator to: {attack:?}");
        let (generator, execution_tx_batch_size) =
            Self::create_generator(accounts, attack.clone(), num_workers);
        Some(Self {
            attack,
            generator,
            execution_tx_batch_size,
        })
    }

    /// Run the generator this many times at once, when producing transactions.
    /// Generators that do little work per transaction should use a larger value.
    pub fn get_execution_batch_size(&self) -> usize {
        self.execution_tx_batch_size
    }

    /// Generate transactions using current generator
    pub fn generate_transactions(&mut self, bank: &Bank) -> (Vec<SanitizedTransaction>, usize) {
        (self.generator)(bank)
    }

    fn create_generator(
        accounts: Arc<AccountsFile>,
        attack: Attack,
        num_workers: usize,
    ) -> (TransactionGenerator, usize) {
        use Attack::*;
        match attack {
            TransferRandom => (transfer_random::generator(accounts, num_workers), 100),
            CreateNonceAccounts => (create_nonce_accounts::generator(accounts, num_workers), 10),
            AllocateRandomLarge => (allocate_random_large::generator(accounts, num_workers), 1),
            AllocateRandomSmall => (allocate_random_small::generator(accounts, num_workers), 10),
            ChainTransactions => (chain_transactions::generator(accounts, num_workers), 10),
            WriteProgram(write_program_config) => (
                write_program::generator(accounts, num_workers, write_program_config),
                1,
            ),
            ReadMaxAccounts => (read_max_accounts::generator(accounts, num_workers), 1),
            WriteMaxAccounts => (write_max_accounts::generator(accounts, num_workers), 1),
            ReadProgram(read_program_config) => (
                read_program::generator(accounts, num_workers, read_program_config),
                1,
            ),
            RecursiveProgram(recursive_program_config) => (
                recursive_program::generator(accounts, num_workers, recursive_program_config),
                1,
            ),
            ColdProgramCache(cold_program_cache_config) => (
                cold_program_cache::generator(accounts, num_workers, cold_program_cache_config),
                1,
            ),
            LargeNop(large_nop_program_config) => (
                large_nop::generator(accounts, num_workers, large_nop_program_config),
                10,
            ),
            TransferRandomWithMemo => (
                transfer_random_with_memo::generator(accounts, num_workers),
                10,
            ),
            ReadNonExistentAccounts => (
                read_non_existent_accounts::generator(accounts, num_workers),
                20,
            ),
        }
    }

    /// Attacks involving expensive computations might be configured with
    /// option to bypass execution. For that, they must be configured to fail
    pub fn use_failed_transaction_hotpath(&self) -> bool {
        match &self.attack {
            Attack::WriteProgram(config)
            | Attack::ReadProgram(config)
            | Attack::RecursiveProgram(config)
            | Attack::ColdProgramCache(config) => config.use_failed_transaction_hotpath,
            _default => false,
        }
    }
}
