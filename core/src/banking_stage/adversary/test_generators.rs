//! Generators for testing banking stage

use {
    super::accounts_file::{AccountsFile, AccountsFileRaw, BlockGeneratorOption},
    crate::{
        banking_stage::consumer::TARGET_NUM_TRANSACTIONS_PER_BATCH,
        validator::{BlockGeneratorAccountsOption, BlockGeneratorConfig},
    },
    block_generator_stress_test::BlockGeneratorStressTestInstruction,
    rand::{seq::SliceRandom, thread_rng, Rng},
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

impl From<BlockGeneratorAccountsOption> for Arc<AccountsFile> {
    fn from(block_generator_config: BlockGeneratorAccountsOption) -> Arc<AccountsFile> {
        match block_generator_config {
            BlockGeneratorAccountsOption::AccountsPath(file_name) => {
                let file_content = std::fs::read_to_string(file_name)
                    .expect("Failed to read the accounts file.\nPath: {file_name}");
                let accounts = serde_json::from_str::<AccountsFileRaw>(&file_content)
                    .expect(
                        "Failed to parse accounts file.\nPath: \
                         {file_name}\nContent:\n{file_content}",
                    )
                    .into();
                Arc::new(accounts)
            }
            BlockGeneratorAccountsOption::Accounts(account_file) => {
                debug!(
                    "Saving accounts for {} starting keypairs into 'payers' group",
                    account_file.payers.len()
                );
                account_file
            }
        }
    }
}

type GeneratorBuilder = fn(accounts: Arc<AccountsFile>, num_workers: usize) -> TransactionGenerator;
struct GeneratorBuilderWithConfig {
    pub builder: GeneratorBuilder,
    /// Run the generator this may times at once, when producing transactions.
    /// Generators that do little work per transaction should use a larger batch size.
    pub batch_size: usize,
}

impl GeneratorBuilderWithConfig {
    fn new(builder: GeneratorBuilder, batch_size: usize) -> Self {
        GeneratorBuilderWithConfig {
            builder,
            batch_size,
        }
    }
}

impl From<BlockGeneratorOption> for GeneratorBuilderWithConfig {
    fn from(val: BlockGeneratorOption) -> Self {
        use BlockGeneratorOption::*;

        match val {
            TransferRandom => Self::new(generator_transfer_random, 100),
            CreateNonceAccounts => Self::new(generator_create_nonce_accounts, 10),
            AllocateRandomLarge => Self::new(generator_allocate_random_large, 1),
            AllocateRandomSmall => Self::new(generator_allocate_random_small, 10),
            ChainTransactions => Self::new(generator_chain_transactions, 10),
            WriteProgram => Self::new(generator_write_program, 1),
        }
    }
}

pub fn get_transaction_generators(
    block_generator_config: BlockGeneratorConfig,
    num_workers: usize,
) -> Vec<(TransactionGenerator, /* tx batches */ usize)> {
    let accounts = Arc::<AccountsFile>::from(block_generator_config.accounts);

    let generators: Vec<GeneratorBuilderWithConfig> = block_generator_config
        .selected_generators
        .iter()
        .map(|generator_option| generator_option.into())
        .collect();

    generators
        .into_iter()
        .map(
            |GeneratorBuilderWithConfig {
                 builder,
                 batch_size,
             }| { (builder(accounts.clone(), num_workers), batch_size) },
        )
        .collect()
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
        assert!(
            accounts.len() >= 2 * BATCH_SIZE,
            "not enough accounts for random transfer generator"
        );

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
        assert!(
            accounts.len() >= BATCH_SIZE,
            "not enough accounts for create nonce account generator"
        );

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
    let set_cu_instruction = ComputeBudgetInstruction::set_compute_unit_limit(25_000);
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
) -> TransactionGenerator {
    let num_payers = accounts.payers.len();
    let program_id = accounts.owner_program_id.expect(
        "Accounts owner program is not specified. Cannot generate write program transactions.",
    );
    let accounts_meta: Vec<AccountMeta> = accounts
        .max_size
        .iter()
        .map(|account| AccountMeta::new(account.pubkey(), false))
        .collect();

    // assumptions that allow to simplify the code
    assert!(
        num_payers != 0,
        "Need at least one payer to pay for the generated transactions"
    );
    if num_payers < TARGET_NUM_TRANSACTIONS_PER_BATCH {
        warn!(
            "Number of payers ({num_payers}) is less than size of the transactions batch \
             ({TARGET_NUM_TRANSACTIONS_PER_BATCH}), this will lead to AccountInUse errors"
        );
    }
    if accounts.max_size.len() % num_payers != 0 {
        info!(
            "Only first {} accounts will be used for write accounts attack",
            accounts.max_size.len() - accounts.max_size.len() % num_payers
        );
    }
    let num_accounts_per_tx = accounts_meta.len() / num_payers;
    let batch_size: usize = TARGET_NUM_TRANSACTIONS_PER_BATCH * num_accounts_per_tx;
    let mut batch_index = 0;
    let num_batches = accounts.max_size.len() / batch_size;

    Box::new(move |bank: &Bank| {
        let payers = accounts.payers.iter().cycle();
        let blockhash = bank.last_blockhash();
        let mut transactions = vec![];

        // splits all the accounts into set of batches
        // which are regularly distributed among workers
        let worker_index = batch_index % num_workers;
        // get current batch
        let batch_begin = batch_index * batch_size;
        let batch_end = batch_begin + batch_size;
        let batch = &accounts_meta[batch_begin..batch_end];

        for (tx_accounts, payer) in batch.chunks(num_accounts_per_tx).zip(payers) {
            let message = create_write_message(payer, &program_id, tx_accounts);
            let transaction = Transaction::new(&[payer], message, blockhash);
            transactions.push(SanitizedTransaction::from_transaction_for_tests(
                transaction,
            ));
        }
        batch_index = (batch_index + 1) % num_batches;
        (transactions, worker_index)
    })
}
