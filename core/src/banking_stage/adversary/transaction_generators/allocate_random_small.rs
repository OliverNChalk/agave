//! Allocates random small accounts.

use {
    super::TransactionGenerator,
    crate::banking_stage::consumer::TARGET_NUM_TRANSACTIONS_PER_BATCH,
    rand::{seq::SliceRandom, thread_rng},
    solana_adversary::accounts_file::AccountsFile,
    solana_keypair::Keypair,
    solana_runtime::bank::Bank,
    solana_system_transaction as system_transaction,
    solana_transaction::sanitized::SanitizedTransaction,
    std::sync::Arc,
};

pub(super) fn generator(accounts: Arc<AccountsFile>, num_workers: usize) -> TransactionGenerator {
    let mut worker_index = 0;
    Box::new(move |bank: &Bank| {
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
