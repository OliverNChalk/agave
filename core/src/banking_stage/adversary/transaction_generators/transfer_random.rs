//! Generates transfers between a set of accounts.

use {
    super::TransactionGenerator,
    crate::banking_stage::consumer::TARGET_NUM_TRANSACTIONS_PER_BATCH,
    rand::{seq::SliceRandom, thread_rng},
    solana_adversary::accounts_file::AccountsFile,
    solana_runtime::bank::Bank,
    solana_signer::Signer,
    solana_system_transaction as system_transaction,
    solana_transaction::sanitized::SanitizedTransaction,
    std::sync::Arc,
};

pub(super) fn generator(accounts: Arc<AccountsFile>, num_workers: usize) -> TransactionGenerator {
    let mut worker_index = 0;
    Box::new(move |bank: &Bank| {
        const BATCH_SIZE: usize = TARGET_NUM_TRANSACTIONS_PER_BATCH;

        let mut transactions = vec![];
        let mut transfer_accounts = accounts
            .payers
            .choose_multiple(&mut thread_rng(), 2 * BATCH_SIZE);
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
