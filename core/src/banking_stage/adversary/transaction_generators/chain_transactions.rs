//! Creates a chain of transactions such that each next transaction depends on the previous one.

use {
    super::TransactionGenerator,
    crate::banking_stage::consumer::TARGET_NUM_TRANSACTIONS_PER_BATCH,
    rand::Rng,
    solana_adversary::accounts_file::AccountsFile,
    solana_runtime::bank::Bank,
    solana_signer::Signer,
    solana_system_transaction as system_transaction,
    solana_transaction::sanitized::SanitizedTransaction,
    std::{iter::zip, sync::Arc},
};

const BATCH_SIZE: usize = TARGET_NUM_TRANSACTIONS_PER_BATCH;

pub fn verify(accounts: &AccountsFile) -> Result<(), String> {
    // We need at least two batches, for the attack to make sense.
    if accounts.payers.len() < 2 * BATCH_SIZE {
        return Err(format!(
            "Not enough `payer` accounts: need at least {}",
            2 * BATCH_SIZE
        ));
    }

    Ok(())
}

pub(super) fn generator(accounts: Arc<AccountsFile>, num_workers: usize) -> TransactionGenerator {
    // Splits all the accounts into set of batches, which are evenly distributed among workers.
    let num_batches = accounts.payers.len() / BATCH_SIZE;
    let mut batch_index = 0;

    Box::new(move |bank: &Bank| {
        let blockhash = bank.last_blockhash();

        // Extra entropy into each transaction to avoid getting already
        // processed errors, but don't let the maximum amount be too high in
        // order to avoid some accounts quickly running out of funds.
        let transfer_amount = rand::thread_rng().gen_range(1..=1000);

        let worker_index = batch_index % num_workers;

        let accounts = &accounts.payers;
        let source_batch = {
            let begin = batch_index * BATCH_SIZE;
            let end = begin + BATCH_SIZE;
            accounts[begin..end].iter()
        };
        let destination_batch = {
            let index = (batch_index + num_workers) % num_batches;
            let begin = index * BATCH_SIZE;
            let end = begin + BATCH_SIZE;
            accounts[begin..end].iter()
        };

        let transactions = zip(source_batch, destination_batch)
            .map(|(source, destination)| {
                system_transaction::transfer(
                    source,
                    &destination.pubkey(),
                    transfer_amount,
                    blockhash,
                )
            })
            .map(SanitizedTransaction::from_transaction_for_tests)
            .collect::<Vec<_>>();

        batch_index = (batch_index + 1) % num_batches;
        (transactions, worker_index)
    })
}
