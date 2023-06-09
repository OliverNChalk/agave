//! Generators for testing banking stage

use {
    crate::{
        banking_stage::consumer::TARGET_NUM_TRANSACTIONS_PER_BATCH,
        validator::{BlockGeneratorAccountsOption, BlockGeneratorConfig, BlockGeneratorOption},
    },
    rand::{seq::SliceRandom, thread_rng},
    serde::Deserialize,
    solana_keypair::Keypair,
    solana_nonce::state as nonce_state,
    solana_rent::Rent,
    solana_runtime::bank::Bank,
    solana_signer::Signer,
    solana_system_interface::instruction as system_instruction,
    solana_system_transaction as system_transaction,
    solana_transaction::{sanitized::SanitizedTransaction, Transaction},
    std::sync::Arc,
};

pub type TransactionGenerator = Box<dyn Send + FnMut(&Bank) -> (Vec<SanitizedTransaction>, usize)>;

#[derive(Default)]
struct AccountsFile {
    payers: Vec<Keypair>,
    _max_size: Vec<Keypair>,
}

impl AccountsFile {
    pub fn with_payers(payers: &[Keypair]) -> Self {
        let payers = payers
            .iter()
            .map(|keypair| keypair.insecure_clone())
            .collect();
        Self {
            payers,
            ..Default::default()
        }
    }
}

impl From<AccountsFileRaw> for AccountsFile {
    fn from(raw: AccountsFileRaw) -> Self {
        let AccountsFileRaw { payers, max_size } = raw;

        let payers = payers.into_iter().map(Into::into).collect();
        let max_size = max_size.into_iter().map(Into::into).collect();

        Self {
            payers,
            _max_size: max_size,
        }
    }
}

#[derive(Deserialize)]
struct AccountsFileRaw {
    #[serde(default)]
    payers: Vec<KeypairRaw>,
    #[serde(default)]
    max_size: Vec<KeypairRaw>,
}

#[derive(Deserialize)]
struct KeypairRaw {
    #[serde(rename = "publicKey")]
    pub _pubkey: String,
    #[serde(rename = "secretKey")]
    pub secret_key: Vec<u8>,
}

impl From<KeypairRaw> for Keypair {
    fn from(raw: KeypairRaw) -> Self {
        assert_eq!(raw.secret_key.len(), 64);
        Self::new_from_array(raw.secret_key[..32].try_into().unwrap())
    }
}

impl From<BlockGeneratorAccountsOption> for AccountsFile {
    fn from(block_generator_config: BlockGeneratorAccountsOption) -> AccountsFile {
        match block_generator_config {
            BlockGeneratorAccountsOption::AccountsPath(file_name) => {
                let file_content = std::fs::read_to_string(file_name)
                    .expect("Failed to read the accounts file.\nPath: {file_name}");
                serde_json::from_str::<AccountsFileRaw>(&file_content)
                    .expect(
                        "Failed to parse accounts file.\nPath: \
                         {file_name}\nContent:\n{file_content}",
                    )
                    .into()
            }
            BlockGeneratorAccountsOption::StartingKeypairs(keypairs) => {
                debug!(
                    "Saving accounts for {} starting keypairs into 'payers' group",
                    keypairs.len()
                );

                AccountsFile::with_payers(&keypairs[..])
            }
        }
    }
}

type GeneratorBuilder = fn(accounts: Arc<AccountsFile>, num_workers: usize) -> TransactionGenerator;
type GeneratorBuilderWithMeta = (GeneratorBuilder, usize);
impl From<BlockGeneratorOption> for GeneratorBuilderWithMeta {
    fn from(val: BlockGeneratorOption) -> Self {
        match val {
            BlockGeneratorOption::TransferRandom => (generator_transfer_random, 100),
            BlockGeneratorOption::CreateNonceAccounts => (generator_create_nonce_accounts, 10),
            BlockGeneratorOption::AllocateRandomLarge => (generator_allocate_random_large, 1),
            BlockGeneratorOption::AllocateRandomSmall => (generator_allocate_random_small, 10),
            BlockGeneratorOption::ChainTransactions => (generator_chain_transactions, 10),
        }
    }
}

pub fn get_transaction_generators(
    block_generator_config: BlockGeneratorConfig,
    num_workers: usize,
) -> Vec<(TransactionGenerator, /* tx batches */ usize)> {
    let accounts = Arc::new(AccountsFile::from(block_generator_config.accounts));

    let generators: Vec<GeneratorBuilderWithMeta> = block_generator_config
        .selected_generators
        .iter()
        .map(|generator_option| generator_option.into())
        .collect();

    generators
        .into_iter()
        .map(|(gen, tx_batches)| (gen(accounts.clone(), num_workers), tx_batches))
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
