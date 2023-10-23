//! A common implementation for generators that operate on single kind of accounts, using a fixed
//! number of these accounts for each transaction in the generated batch.
//!
//! Accounts are provided using an instance of [`Cycler`].
//!
//! It is also expected that a single account from the `payers` list will be used to pay for every
//! transaction in a round robin fashion.

use {
    crate::banking_stage::{
        adversary::generator_components::{
            cycler::{self, Cycler},
            IndexByModulo,
        },
        consumer::TARGET_NUM_TRANSACTIONS_PER_BATCH,
    },
    solana_adversary::accounts_file::AccountsFile,
    solana_keypair::Keypair,
    solana_runtime::bank::Bank,
    solana_transaction::{sanitized::SanitizedTransaction, Transaction},
    std::sync::Arc,
};

const BATCH_SIZE: usize = TARGET_NUM_TRANSACTIONS_PER_BATCH;

/// Checks the `payers` accounts to hold enough payers to pay for the
/// `TRANSACTION_LEVEL_STACK_HEIGHT` transactions in parallel.
///
/// For the target accounts, it needs to know the total number of them and the number of account per
/// transaction, to verify that there would be enough, to populate the whole batch with different
/// accounts.
pub(crate) fn verify(
    accounts: &AccountsFile,
    target_accounts_name: &'static str,
    target_accounts_len: usize,
    tx_accounts: usize,
) -> Result<(), String> {
    let payers_len = accounts.payers.len();

    if payers_len < BATCH_SIZE {
        return Err(format!(
            "Not enough `payer` accounts: need at least {BATCH_SIZE}\n`payer` accounts: \
             {payers_len}"
        ));
    }

    let accounts_per_batch = tx_accounts * BATCH_SIZE;

    if target_accounts_len < accounts_per_batch {
        return Err(format!(
            "Not enough `{target_accounts_name}` accounts: need at least \
             {accounts_per_batch}\n`{target_accounts_name}` accounts: {target_accounts_len}"
        ));
    }

    Ok(())
}

pub(crate) fn generator<AccountsHolder, PopulateTransaction>(
    accounts: Arc<AccountsFile>,
    mut target_accounts: Cycler<AccountsHolder, Keypair>,
    num_workers: usize,
    mut populate_transaction: PopulateTransaction,
) -> impl FnMut(usize, &Bank) -> (Vec<SanitizedTransaction>, usize) + Send
where
    AccountsHolder: 'static + Send,
    PopulateTransaction:
        FnMut(&Bank, &Keypair, cycler::ChunkIter<'_, '_, Keypair>) -> Transaction + Send + 'static,
{
    let mut worker_index = IndexByModulo::new(num_workers);

    let mut payers = Cycler::over(accounts, |accounts| accounts.payers.iter());

    move |tx_accounts: usize, bank: &Bank| {
        let transactions = (0..BATCH_SIZE)
            .map(|_batch_idx| {
                let payer = payers.take_one();

                target_accounts.with_chunk(tx_accounts, |target_accounts| {
                    populate_transaction(bank, payer, target_accounts)
                })
            })
            .map(SanitizedTransaction::from_transaction_for_tests)
            .collect();

        (transactions, worker_index.next())
    }
}
