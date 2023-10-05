//! Creates a chain of transactions such that each next transaction depends on the previous one.

use {
    super::TransactionGenerator, crate::banking_stage::consumer::TARGET_NUM_TRANSACTIONS_PER_BATCH,
    solana_adversary::accounts_file::AccountsFile, solana_runtime::bank::Bank,
    solana_signer::Signer, solana_system_transaction as system_transaction,
    solana_transaction::sanitized::SanitizedTransaction, std::sync::Arc,
};

pub(super) fn generator(accounts: Arc<AccountsFile>, num_workers: usize) -> TransactionGenerator {
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
