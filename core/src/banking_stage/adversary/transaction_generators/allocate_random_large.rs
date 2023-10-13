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

// Large allocations are expensive, so keep batch size small.
const BATCH_SIZE: usize = 1;

pub fn verify(accounts: &AccountsFile) -> Result<(), String> {
    if accounts.payers.len() < BATCH_SIZE {
        return Err(format!(
            "Not enough `payer` accounts: need at least {BATCH_SIZE}",
        ));
    }

    Ok(())
}

pub(super) fn generator(accounts: Arc<AccountsFile>, num_workers: usize) -> TransactionGenerator {
    const ACCOUNT_SIZE: u64 = MAX_PERMITTED_DATA_LENGTH;

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
