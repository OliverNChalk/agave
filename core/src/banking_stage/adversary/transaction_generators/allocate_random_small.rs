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

const BATCH_SIZE: usize = TARGET_NUM_TRANSACTIONS_PER_BATCH;

pub fn verify(accounts: &AccountsFile) -> Result<(), String> {
    if accounts.payers.len() < BATCH_SIZE {
        return Err(format!(
            "Not enough `payer` accounts: need at least {BATCH_SIZE}",
        ));
    }

    Ok(())
}

pub(super) fn generator(accounts: Arc<AccountsFile>, num_workers: usize) -> TransactionGenerator {
    const ACCOUNT_SIZE: u64 = 1;

    let mut worker_index = 0;
    Box::new(move |bank: &Bank| {
        let blockhash = bank.last_blockhash();

        let transactions = accounts
            .payers
            .choose_multiple(&mut thread_rng(), BATCH_SIZE)
            .map(|account| {
                system_transaction::allocate(account, &Keypair::new(), blockhash, ACCOUNT_SIZE)
            })
            .map(SanitizedTransaction::from_transaction_for_tests)
            .collect::<Vec<_>>();

        let current_worker_index = worker_index;
        worker_index = (worker_index + 1) % num_workers;
        (transactions, current_worker_index)
    })
}
