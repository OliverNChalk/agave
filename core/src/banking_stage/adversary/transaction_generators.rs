//! Generators for testing banking stage

use {
    crate::banking_stage::consumer::TARGET_NUM_TRANSACTIONS_PER_BATCH,
    block_generator_stress_test::BlockGeneratorStressTestInstruction,
    rand::{seq::SliceRandom, thread_rng, Rng},
    solana_adversary::{
        accounts_file::AccountsFile,
        adversary_feature_set::{
            replay_stage_attack,
            replay_stage_attack::{Attack, WriteProgramConfig},
        },
        block_generator_config::BlockGeneratorConfig,
    },
    solana_compute_budget_interface::ComputeBudgetInstruction,
    solana_instruction::{AccountMeta, Instruction},
    solana_keypair::Keypair,
    solana_message::Message,
    solana_nonce::state as nonce_state,
    solana_pubkey::Pubkey,
    solana_rent::Rent,
    solana_runtime::bank::Bank,
    solana_signer::Signer,
    solana_system_interface::instruction as system_instruction,
    solana_system_transaction as system_transaction,
    solana_transaction::{sanitized::SanitizedTransaction, Transaction},
    std::sync::Arc,
};

pub type TransactionGenerator = Box<dyn Send + FnMut(&Bank) -> (Vec<SanitizedTransaction>, usize)>;

/// Encapsulate logic for managing selected generator
pub struct ActiveGenerator {
    num_workers: usize,
    accounts: Arc<AccountsFile>,
    current: Option<(TransactionGenerator, /* execution tx batches */ usize)>,
    // used to simplify check if the selected_generator should be changed
    current_attack: Option<replay_stage_attack::Attack>,
}

impl ActiveGenerator {
    pub fn new(block_generator_config: BlockGeneratorConfig, num_workers: usize) -> Self {
        let accounts = Arc::<AccountsFile>::from(block_generator_config.accounts);
        Self {
            num_workers,
            accounts,
            current: None,
            current_attack: None,
        }
    }

    /// Run the generator this many times at once, when producing transactions.
    /// Generators that do little work per transaction should use a larger value.
    pub fn get_num_generator_exec_batch_size(&self) -> usize {
        self.current
            .as_ref()
            .map(|generator| generator.1)
            .unwrap_or(0)
    }

    /// Generate transactions using current generator
    pub fn generate_transactions(&mut self, bank: &Bank) -> (Vec<SanitizedTransaction>, usize) {
        self.current
            .as_mut()
            .map(|generator| (generator.0)(bank))
            .unwrap_or((vec![], 0))
    }

    fn create_generator(&self) -> Option<(TransactionGenerator, usize)> {
        let Self {
            accounts,
            num_workers,
            current_attack,
            ..
        } = self;
        let accounts = accounts.clone();

        use Attack::*;
        current_attack.as_ref().map(|attack| match attack {
            TransferRandom => (generator_transfer_random(accounts, *num_workers), 100),
            CreateNonceAccounts => (generator_create_nonce_accounts(accounts, *num_workers), 10),
            AllocateRandomLarge => (generator_allocate_random_large(accounts, *num_workers), 1),
            AllocateRandomSmall => (generator_allocate_random_small(accounts, *num_workers), 10),
            ChainTransactions => (generator_chain_transactions(accounts, *num_workers), 10),
            WriteProgram(write_program_config) => (
                generator_write_program(accounts, *num_workers, write_program_config.clone()),
                1,
            ),
        })
    }

    /// Create a new generator using existing accounts and reset the state of the structure
    pub fn ensure_active(&mut self, attack: Option<replay_stage_attack::Attack>) {
        if self.current_attack == attack {
            return;
        }
        info!("Reset selected generator to: {attack:?}");
        self.current_attack = attack.clone();
        self.current = self.create_generator();
    }

    /// Check if any generator has been selected
    pub fn is_active(&self) -> bool {
        self.current.is_some()
    }

    /// Attacks involving expensive computations might be configured with
    /// option to bypass execution. For that, they must be configured to fail
    pub fn use_failed_transaction_hotpath(&self) -> bool {
        if let Some(replay_stage_attack::Attack::WriteProgram(write_attack_config)) =
            &self.current_attack
        {
            write_attack_config.use_failed_transaction_hotpath
        } else {
            false
        }
    }
}

/// Generates transfers between a set of accounts.
fn generator_transfer_random(
    accounts: Arc<AccountsFile>,
    num_workers: usize,
) -> TransactionGenerator {
    let mut worker_index = 0;
    Box::new(move |bank: &Bank| {
        const BATCH_SIZE: usize = TARGET_NUM_TRANSACTIONS_PER_BATCH;
        let accounts = &accounts.payers;

        let mut transactions = vec![];
        let mut transfer_accounts = accounts.choose_multiple(&mut thread_rng(), 2 * BATCH_SIZE);
        for _ in 0..BATCH_SIZE {
            let transaction = system_transaction::transfer(
                transfer_accounts.next().unwrap(),
                &transfer_accounts.next().unwrap().pubkey(),
                1,
                bank.last_blockhash(),
            );
            transactions.push(SanitizedTransaction::from_transaction_for_tests(
                transaction,
            ));
        }

        let current_worker_index = worker_index;
        worker_index = (worker_index + 1) % num_workers;
        (transactions, current_worker_index)
    })
}

/// Generates nonce accounts.
fn generator_create_nonce_accounts(
    accounts: Arc<AccountsFile>,
    num_workers: usize,
) -> TransactionGenerator {
    let mut worker_index = 0;
    Box::new(move |bank: &Bank| {
        const BATCH_SIZE: usize = TARGET_NUM_TRANSACTIONS_PER_BATCH;
        let balance = Rent::default().minimum_balance(nonce_state::State::size());

        let accounts = &accounts.payers;
        let mut transactions = vec![];
        let mut payers = accounts.choose_multiple(&mut thread_rng(), BATCH_SIZE);
        for _ in 0..BATCH_SIZE {
            let payer = payers.next().unwrap();
            let nonce_account = Keypair::new();
            let instr = system_instruction::create_nonce_account(
                &payer.pubkey(),
                &nonce_account.pubkey(),
                &payer.pubkey(), // Make the fee payer the nonce account authority
                balance,
            );
            let mut tx_create_nonce_account =
                Transaction::new_with_payer(&instr, Some(&payer.pubkey()));

            tx_create_nonce_account
                .try_sign(&[&nonce_account, payer], bank.last_blockhash())
                .unwrap();

            transactions.push(SanitizedTransaction::from_transaction_for_tests(
                tx_create_nonce_account,
            ));
        }

        let current_worker_index = worker_index;
        worker_index = (worker_index + 1) % num_workers;
        (transactions, current_worker_index)
    })
}

/// Allocates random large accounts.
fn generator_allocate_random_large(
    accounts: Arc<AccountsFile>,
    num_workers: usize,
) -> TransactionGenerator {
    let mut worker_index = 0;
    Box::new(move |bank: &Bank| {
        const BATCH_SIZE: usize = 1;
        const ACCOUNT_SIZE: u64 = solana_system_interface::MAX_PERMITTED_DATA_LENGTH;

        let accounts = &accounts.payers;

        let mut transactions = vec![];

        let mut accounts = accounts.choose_multiple(&mut thread_rng(), BATCH_SIZE);
        for _ in 0..BATCH_SIZE {
            let transaction = system_transaction::allocate(
                accounts.next().unwrap(),
                &Keypair::new(),
                bank.last_blockhash(),
                ACCOUNT_SIZE,
            );
            transactions.push(SanitizedTransaction::from_transaction_for_tests(
                transaction,
            ));
        }

        let current_worker_index = worker_index;
        worker_index = (worker_index + 1) % num_workers;
        (transactions, current_worker_index)
    })
}

/// Allocates random small accounts.
fn generator_allocate_random_small(
    accounts: Arc<AccountsFile>,
    num_workers: usize,
) -> TransactionGenerator {
    let mut worker_index = 0;
    Box::new(move |bank: &Bank| {
        // Large allocations are expensive, so keep batch size small.
        const BATCH_SIZE: usize = TARGET_NUM_TRANSACTIONS_PER_BATCH;
        const ACCOUNT_SIZE: u64 = 1;

        let accounts = &accounts.payers;

        let mut transactions = vec![];

        let mut accounts = accounts.choose_multiple(&mut thread_rng(), BATCH_SIZE);
        for _ in 0..BATCH_SIZE {
            let transaction = system_transaction::allocate(
                accounts.next().unwrap(),
                &Keypair::new(),
                bank.last_blockhash(),
                ACCOUNT_SIZE,
            );
            transactions.push(SanitizedTransaction::from_transaction_for_tests(
                transaction,
            ));
        }

        let current_worker_index = worker_index;
        worker_index = (worker_index + 1) % num_workers;
        (transactions, current_worker_index)
    })
}

/// Creates a circular chain of transactions such that each next transaction depends on the previous
fn generator_chain_transactions(
    accounts: Arc<AccountsFile>,
    num_workers: usize,
) -> TransactionGenerator {
    let mut batch_index = 0;
    Box::new(move |bank: &Bank| {
        const BATCH_SIZE: usize = TARGET_NUM_TRANSACTIONS_PER_BATCH;
        const TRANSFER_AMOUNT: u64 = 1;

        let accounts = &accounts.payers;

        let blockhash = bank.last_blockhash();
        let mut transactions = vec![];

        // splits all the accounts into set of batches
        // which are regularly distributed among workers
        let num_batches = accounts.len() / BATCH_SIZE;
        let worker_index = batch_index % num_workers;
        // get current batch
        let batch_begin = batch_index * BATCH_SIZE;
        let batch_end = batch_begin + BATCH_SIZE;
        let batch = &accounts[batch_begin..batch_end];

        // get next batch
        let next_batch_index = (batch_index + num_workers) % num_batches;
        let next_batch_begin = next_batch_index * BATCH_SIZE;
        let next_batch_end = next_batch_begin + BATCH_SIZE;
        let next_batch = &accounts[next_batch_begin..next_batch_end];
        for (source_account, dest_account) in batch.iter().zip(next_batch.iter()) {
            let transaction = system_transaction::transfer(
                source_account,
                &dest_account.pubkey(),
                TRANSFER_AMOUNT,
                blockhash,
            );
            transactions.push(SanitizedTransaction::from_transaction_for_tests(
                transaction,
            ));
        }

        batch_index = (batch_index + 1) % num_batches;
        (transactions, worker_index)
    })
}

fn create_write_message(
    payer: &Keypair,
    program_id: &Pubkey,
    accounts_meta: &[AccountMeta],
    transaction_cu_budget: u32,
) -> Message {
    let rnd = thread_rng().gen_range(0..=u64::MAX);
    let data = BlockGeneratorStressTestInstruction::WriteAccounts {
        value: 128,
        random: rnd,
    };
    // Explicitly specify the CU budget to avoid dropping some transactions on the CU check side
    // The constraint is that 48M/64 > CU limit
    // To maximize number of txs in the block, it is beneficial to set CU limit to be close to the real CU consumption
    // Default is 200k
    let set_cu_instruction =
        ComputeBudgetInstruction::set_compute_unit_limit(transaction_cu_budget);
    let write_instruction = Instruction::new_with_borsh(*program_id, &data, accounts_meta.to_vec());
    Message::new(
        &[set_cu_instruction, write_instruction],
        Some(&payer.pubkey()),
    )
}

/// Creates a generator that executes the program
fn generator_write_program(
    accounts: Arc<AccountsFile>,
    num_workers: usize,
    config: WriteProgramConfig,
) -> TransactionGenerator {
    // To enforce each transaction within the batch to be paid by a new payer
    // This is to reduce AccountsInUse errors
    let num_payers = accounts.payers.len();
    if num_payers < config.transaction_batch_size * num_workers {
        warn!(
            "Number of payers ({num_payers} is less than number of workers by batch size \
             ({num_workers} x {}).This will lead to AccountInUse errors.",
            config.transaction_batch_size
        );
    }
    let program_id = accounts
        .owner_program_id
        .expect("`owner_program_id` presense is checked during the config validation");
    let num_max_accounts = accounts.max_size.len();
    let accounts_meta: Vec<AccountMeta> = accounts
        .max_size
        .iter()
        .map(|account| AccountMeta::new(account.pubkey(), false))
        .collect();

    let accounts_batch_size: usize = config.transaction_batch_size * config.num_accounts_per_tx;

    let mut batch_index = 0;
    // having index allows to use new payer for each run of the closure
    // it is impossible to use cyclic iterator because it would reference payers
    let mut payer_index = 0;
    let num_batches = num_max_accounts / accounts_batch_size;
    Box::new(move |bank: &Bank| {
        // splits all the accounts into set of batches
        // which are evenly distributed among workers
        let worker_index = batch_index % num_workers;
        // get current batch
        let accounts_batch = {
            let batch_begin = batch_index * accounts_batch_size;
            let batch_end = batch_begin + accounts_batch_size;
            &accounts_meta[batch_begin..batch_end]
        };

        let blockhash = bank.last_blockhash();
        let mut transactions = Vec::with_capacity(config.transaction_batch_size);
        for tx_accounts in accounts_batch.chunks(config.num_accounts_per_tx) {
            let payer = &accounts.payers[payer_index];
            let message = create_write_message(
                payer,
                &program_id,
                tx_accounts,
                config.transaction_cu_budget,
            );
            let transaction = Transaction::new(&[payer], message, blockhash);
            transactions.push(SanitizedTransaction::from_transaction_for_tests(
                transaction,
            ));
            payer_index = (payer_index + 1) % num_payers;
        }
        batch_index = (batch_index + 1) % num_batches;
        (transactions, worker_index)
    })
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        solana_ledger::genesis_utils::GenesisConfigInfo,
        solana_runtime::{bank::Bank, genesis_utils::create_genesis_config},
    };

    fn create_test_bank() -> Arc<Bank> {
        let GenesisConfigInfo { genesis_config, .. } = create_genesis_config(10_000);
        Bank::new_no_wallclock_throttle_for_tests(&genesis_config).0
    }

    #[test]
    fn test_generator_write_program() {
        let num_workers = 1;
        let num_payers = 64;
        let num_max_sized_accounts = 1024;
        let owner_program_id = Pubkey::default();
        let payers_accounts: Vec<Keypair> = (0..num_payers).map(|_| Keypair::new()).collect();
        let max_size_accounts: Vec<Keypair> = (0..num_max_sized_accounts)
            .map(|_| Keypair::new())
            .collect();
        let accounts = Arc::new(AccountsFile::with_payers_and_max_size(
            &owner_program_id,
            &payers_accounts,
            &max_size_accounts,
        ));
        let config = WriteProgramConfig {
            transaction_batch_size: 32,
            num_accounts_per_tx: 8,
            transaction_cu_budget: 5000,
            use_failed_transaction_hotpath: false,
        };
        let mut generate = generator_write_program(accounts, num_workers, config.clone());

        let bank = create_test_bank();

        let (txs, _worker_id) = generate(&bank);

        assert_eq!(txs.len(), config.transaction_batch_size);

        // beside of accounts to be modified, tx includes also payer, owner program id, compute budget program id
        let expected_num_accounts_per_tx = config.num_accounts_per_tx + 3;
        for tx in txs {
            assert_eq!(
                tx.message().account_keys().len(),
                expected_num_accounts_per_tx
            );
            let instructions = &tx.message().instructions();
            assert_eq!(instructions.len(), 2);

            let mut ix_iter = tx.message().program_instructions_iter();
            ix_iter.next();
            assert_eq!(
                ix_iter.next().map(|(program_id, _ix)| program_id),
                Some(&owner_program_id)
            );
        }
    }
}
