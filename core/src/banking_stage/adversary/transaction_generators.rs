//! Generators for testing banking stage

use {
    rayon::ThreadPool,
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
pub(super) mod cpi_program;
pub(super) mod create_nonce_accounts;
pub(super) mod large_nop;
pub(super) mod read_max_size_accounts;
pub(super) mod read_non_existent_accounts;
pub(super) mod read_program;
pub(super) mod recursive_program;
pub(super) mod transfer_random;
pub(super) mod transfer_random_with_memo;
pub(super) mod write_max_size_accounts;
pub(super) mod write_program;

/// Attacks are essentially sequences of transactions.  Each will have a function that is
/// responsible for generating these transactions, and those functions will adhere to this type.
pub type TransactionGenerator = Box<dyn Send + FnMut(&Bank) -> (Vec<SanitizedTransaction>, usize)>;

/// Encapsulate logic for managing selected generator
pub struct ActiveGenerator {
    attack: Attack,
    generator: TransactionGenerator,
}

impl ActiveGenerator {
    /// Produces an `ActiveGenerator` if the `SelectedReplayAttack` indicates an attack needs to be
    /// run.
    pub fn given(
        selected_attack: SelectedReplayAttack,
        num_workers: usize,
        tx_generator_thread_pool: Arc<ThreadPool>,
    ) -> Option<Self> {
        let (accounts, attack) = match selected_attack {
            SelectedReplayAttack::Selected { accounts, attack } => (accounts, attack),
            SelectedReplayAttack::None => return None,
        };

        info!("Reset selected generator to: {attack:?}");
        let generator = Self::create_generator(
            accounts,
            attack.clone(),
            num_workers,
            tx_generator_thread_pool,
        );
        Some(Self { attack, generator })
    }

    /// Generate transactions using current generator
    pub fn generate_transactions(&mut self, bank: &Bank) -> (Vec<SanitizedTransaction>, usize) {
        (self.generator)(bank)
    }

    fn create_generator(
        accounts: Arc<AccountsFile>,
        attack: Attack,
        num_workers: usize,
        tx_generator_thread_pool: Arc<ThreadPool>,
    ) -> TransactionGenerator {
        use Attack::*;
        match attack {
            TransferRandom => transfer_random::generator(accounts, num_workers),
            CreateNonceAccounts => create_nonce_accounts::generator(accounts, num_workers),
            AllocateRandomLarge => allocate_random_large::generator(accounts, num_workers),
            AllocateRandomSmall => allocate_random_small::generator(accounts, num_workers),
            ChainTransactions => chain_transactions::generator(accounts, num_workers),
            WriteProgram(write_program_config) => {
                write_program::generator(accounts, num_workers, write_program_config)
            }
            ReadMaxSizeAccounts => read_max_size_accounts::generator(accounts, num_workers),
            WriteMaxSizeAccounts => write_max_size_accounts::generator(accounts, num_workers),
            ReadProgram(read_program_config) => {
                read_program::generator(accounts, num_workers, read_program_config)
            }
            RecursiveProgram(recursive_program_config) => {
                recursive_program::generator(accounts, num_workers, recursive_program_config)
            }
            CpiProgram(cpi_program_config) => {
                cpi_program::generator(accounts, num_workers, cpi_program_config)
            }
            ColdProgramCache(cold_program_cache_config) => {
                cold_program_cache::generator(accounts, num_workers, cold_program_cache_config)
            }
            LargeNop(large_nop_program_config) => large_nop::generator(
                accounts,
                num_workers,
                large_nop_program_config,
                tx_generator_thread_pool,
            ),
            TransferRandomWithMemo => transfer_random_with_memo::generator(accounts, num_workers),
            ReadNonExistentAccounts(non_existent_accounts_config) => {
                read_non_existent_accounts::generator(
                    accounts,
                    num_workers,
                    non_existent_accounts_config,
                    tx_generator_thread_pool,
                )
            }
        }
    }

    /// Attacks involving expensive computations might be configured with
    /// option to bypass execution. For that, they must be configured to fail
    pub fn use_failed_transaction_hotpath(&self) -> bool {
        match &self.attack {
            Attack::WriteProgram(config)
            | Attack::ReadProgram(config)
            | Attack::RecursiveProgram(config)
            | Attack::CpiProgram(config)
            | Attack::ColdProgramCache(config) => config.use_failed_transaction_hotpath,
            Attack::ReadNonExistentAccounts(config) => config.use_failed_transaction_hotpath,
            _default => false,
        }
    }

    /// Attacks that want the fee payer to be an invalid account should set this flag.  Invalidator
    /// will then avoid both loading and charging the fee payer account.
    pub fn use_invalid_fee_payer(&self) -> bool {
        match &self.attack {
            Attack::ReadNonExistentAccounts(config) => config.use_invalid_fee_payer,
            _default => false,
        }
    }
}
