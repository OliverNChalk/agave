//! A common implementation for transaction generators that want to put accounts of the maximum
//! possible size ([`MAX_PERMITTED_DATA_LENGTH`]) into a transaction and put as many of those, as
//! possible.  The limiting factor here is the limit on the data that a single transaction can load
//! ([`MAX_LOADED_ACCOUNTS_DATA_SIZE_BYTES`]).
//!
//! Delegates account selection to [`rotate_accounts`].

use {
    crate::banking_stage::adversary::{
        generator_components::{cycler, Cycler},
        generator_templates::rotate_accounts,
    },
    solana_adversary::accounts_file::AccountsFile,
    solana_compute_budget::compute_budget_limits::MAX_LOADED_ACCOUNTS_DATA_SIZE_BYTES,
    solana_keypair::Keypair,
    solana_runtime::bank::Bank,
    solana_system_interface::MAX_PERMITTED_DATA_LENGTH,
    solana_transaction::{sanitized::SanitizedTransaction, Transaction},
    std::sync::Arc,
};

// There is a limit on the amount of account data that can be loaded in a single
// transaction. This represents the number of maximum sized accounts that can be
// loaded up with a single tx.
pub const TX_MAX_NUM_MAX_SIZE_ACCOUNTS: usize =
    MAX_LOADED_ACCOUNTS_DATA_SIZE_BYTES.get() as usize / MAX_PERMITTED_DATA_LENGTH as usize;

pub(crate) fn verify(accounts: &AccountsFile) -> Result<(), String> {
    rotate_accounts::verify(
        accounts,
        "max_size",
        accounts.max_size.len(),
        TX_MAX_NUM_MAX_SIZE_ACCOUNTS,
    )
}

pub(crate) fn generator<PopulateTransaction>(
    accounts: Arc<AccountsFile>,
    num_workers: usize,
    populate_transaction: PopulateTransaction,
) -> impl FnMut(&Bank) -> (Vec<SanitizedTransaction>, usize) + Send
where
    PopulateTransaction:
        FnMut(&Bank, &Keypair, cycler::ChunkIter<'_, '_, Keypair>) -> Transaction + Send + 'static,
{
    let target_accounts = Cycler::over(accounts.clone(), |accounts| accounts.max_size.iter());

    let mut rotate =
        rotate_accounts::generator(accounts, target_accounts, num_workers, populate_transaction);

    move |bank: &Bank| rotate(TX_MAX_NUM_MAX_SIZE_ACCOUNTS, bank)
}
