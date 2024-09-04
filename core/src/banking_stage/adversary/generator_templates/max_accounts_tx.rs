//! A common implementation for transaction generators that want to put as many account references
//! into a transaction as possible.  The limit is defined by the transaction size.
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
    solana_packet::PACKET_DATA_SIZE,
    solana_pubkey::PUBKEY_BYTES,
    solana_runtime::bank::Bank,
    solana_system_interface::MAX_PERMITTED_DATA_LENGTH,
    solana_transaction::{
        sanitized::{SanitizedTransaction, MAX_TX_ACCOUNT_LOCKS},
        Transaction,
    },
    std::sync::Arc,
};

// Space in bytes consumed by a transaction with a single payer and no instructions.
//
// Correctness of this value is checked in [`tests::check_tx_padding`].
const TX_WITH_SINGLE_PAYER_SIZE_BYTES: usize = 134;

// See how many account addresses can we pack into a single transaction space.  One account is
// the payer, the rest are our "transaction accounts".
pub const TX_MAX_ATTACK_ACCOUNTS_IN_PACKET: usize =
    (PACKET_DATA_SIZE - TX_WITH_SINGLE_PAYER_SIZE_BYTES) / PUBKEY_BYTES;

// There is a limit on the amount of account data that can be loaded in a single
// transaction. This represents the number of maximum sized accounts that can be
// loaded up with a single tx.
pub const TX_MAX_NUM_MAX_SIZE_ACCOUNTS: usize =
    (MAX_LOADED_ACCOUNTS_DATA_SIZE_BYTES.get() as u64 / MAX_PERMITTED_DATA_LENGTH) as usize;

// For cases where we do not have a bank yet, use `MAX_TX_ACCOUNT_LOCKS` as an
// optimistic approximation.
//
// Network should be configured with this value or something smaller.  So in
// case of a mismatch, it is possible we are sending more accounts than can
// actually be locked and tx may fail.
//
// Subtract 1 for the fee payer account.
const TX_MAX_ATTACK_ACCOUNT_LOCKS: usize = MAX_TX_ACCOUNT_LOCKS - 1;

pub(crate) fn verify(
    accounts: &AccountsFile,
    target_accounts_name: &'static str,
    target_accounts_len: usize,
) -> Result<(), String> {
    // Take the lowest value given the following tx constraints:
    // 1. packet data size
    // 2. tx account data load limit
    // 3. tx account lock limit
    let tx_accounts = TX_MAX_NUM_MAX_SIZE_ACCOUNTS
        .min(TX_MAX_ATTACK_ACCOUNTS_IN_PACKET.min(TX_MAX_ATTACK_ACCOUNT_LOCKS));

    rotate_accounts::verify(
        accounts,
        target_accounts_name,
        target_accounts_len,
        tx_accounts,
    )
}

// [`verify()`] does not have access to a valid bank, and so it cannot call
// [`Bank::get_transaction_account_lock_limit()`].  This function does this
// check again, but during the generator execution, panicking if the limit was
// indeed exceeded.
fn verify_tx_max_accounts_during_generation(
    bank: &Bank,
    target_accounts_name: &'static str,
    tx_accounts: usize,
) {
    let account_lock_limit = bank.get_transaction_account_lock_limit();

    if tx_accounts <= account_lock_limit {
        return;
    }

    // Transactions are expected to fail due to exceeding account lock limit... let's panic.
    if account_lock_limit >= MAX_TX_ACCOUNT_LOCKS {
        panic!("`tx_max_accounts` passed into `generator()` exceeds one passed into `verify()`");
    } else {
        panic!(
            "`bank.get_transaction_account_lock_limit()` is below \
             `MAX_TX_ACCOUNT_LOCKS`.\nbank.get_transaction_account_lock_limit(): \
             {account_lock_limit}\nMAX_TX_ACCOUNT_LOCKS: {MAX_TX_ACCOUNT_LOCKS}\n`verify()` was \
             not able to check that the number of `{target_accounts_name}` accounts was \
             constrained enough.",
        );
    }
}

pub(crate) fn generator<AccountsHolder, PopulateTransaction>(
    accounts: Arc<AccountsFile>,
    target_accounts_name: &'static str,
    target_accounts: Cycler<AccountsHolder, Keypair>,
    num_workers: usize,
    populate_transaction: PopulateTransaction,
) -> impl FnMut(&Bank) -> (Vec<SanitizedTransaction>, usize) + Send
where
    AccountsHolder: 'static + Send,
    PopulateTransaction:
        FnMut(&Bank, &Keypair, cycler::ChunkIter<'_, '_, Keypair>) -> Transaction + Send + 'static,
{
    let mut rotate =
        rotate_accounts::generator(accounts, target_accounts, num_workers, populate_transaction);

    move |bank: &Bank| {
        let tx_accounts = TX_MAX_NUM_MAX_SIZE_ACCOUNTS.min(TX_MAX_ATTACK_ACCOUNTS_IN_PACKET.min(
            // The very first account will be our payer, while the rest can be populated by our
            // payload.
            bank.get_transaction_account_lock_limit().saturating_sub(1),
        ));

        verify_tx_max_accounts_during_generation(bank, target_accounts_name, tx_accounts);

        rotate(tx_accounts, bank)
    }
}

#[cfg(test)]
mod tests {
    use {
        super::TX_WITH_SINGLE_PAYER_SIZE_BYTES, bincode, solana_hash::Hash,
        solana_keypair::Keypair, solana_message::legacy::Message as LegacyMessage,
        solana_signer::Signer, solana_transaction::Transaction,
    };

    #[test]
    fn check_tx_padding() {
        let keypair = Keypair::new();
        let payer = keypair.pubkey();
        let blockhash = Hash::new_unique();

        let message =
            LegacyMessage::new_with_compiled_instructions(1, 0, 0, vec![payer], blockhash, vec![]);

        let tx = Transaction::new(&[&keypair], message, blockhash);
        let bytes = bincode::serialize(&tx).unwrap();

        assert_eq!(
            bytes.len(),
            TX_WITH_SINGLE_PAYER_SIZE_BYTES,
            "`TX_WITH_SINGLE_PAYER_SIZE_BYTES` should be set to the size of an empty tx with a \
             single payer"
        );
    }
}
