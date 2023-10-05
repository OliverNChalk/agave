//! Allocates random large accounts.

use {
    super::TransactionGenerator,
    rand::{seq::SliceRandom, thread_rng},
    solana_adversary::accounts_file::AccountsFile,
    solana_keypair::Keypair,
    solana_runtime::bank::Bank,
    solana_system_interface::MAX_PERMITTED_DATA_LENGTH,
    solana_system_transaction as system_transaction,
    solana_transaction::sanitized::SanitizedTransaction,
    std::sync::Arc,
};

pub(super) fn generator(accounts: Arc<AccountsFile>, num_workers: usize) -> TransactionGenerator {
    let mut worker_index = 0;
    Box::new(move |bank: &Bank| {
        // Large allocations are expensive, so keep batch size small.
        const BATCH_SIZE: usize = 1;
        const ACCOUNT_SIZE: u64 = MAX_PERMITTED_DATA_LENGTH;

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
