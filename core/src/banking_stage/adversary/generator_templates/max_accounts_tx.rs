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
    solana_keypair::Keypair,
    solana_packet::PACKET_DATA_SIZE,
    solana_pubkey::PUBKEY_BYTES,
    solana_runtime::bank::Bank,
    solana_transaction::{
        sanitized::{SanitizedTransaction, MAX_TX_ACCOUNT_LOCKS},
        Transaction,
    },
    std::sync::Arc,
};

// Some space in the transaction is used by transaction headers and other data.  It is estimated by
// the `TX_PADDING` value.  This is for the legacy transactions for now.
//
// Correctness of this value is checked in [`tests::check_tx_padding`].
const TX_PADDING: usize = 69;

// See how many account addresses can we pack into a single transaction space.  One account is
// the payer, the rest are our "transaction accounts".
pub(crate) const TX_MAX_ATTACK_ACCOUNTS_IN_PACKET: usize =
    (PACKET_DATA_SIZE - TX_PADDING) / PUBKEY_BYTES - 1;

pub(crate) fn verify(
    accounts: &AccountsFile,
    target_accounts_name: &'static str,
    target_accounts_len: usize,
) -> Result<(), String> {
    // If the network is configured with an account limit lower than the one based on the packet
    // size, we should use the lower value.
    let tx_accounts = TX_MAX_ATTACK_ACCOUNTS_IN_PACKET.min(
        // We can not call `bank.get_transaction_account_lock_limit()` as we do not have a bank yet.
        // Use `MAX_TX_ACCOUNT_LOCKS` as pessimistic approximation.
        //
        // Network should be configured with this value or something smaller.  So in case of a miss
        // match, it is more likely that we just request more accounts than necessary, rather then
        // miss the check altogether.
        MAX_TX_ACCOUNT_LOCKS.saturating_sub(1),
    );

    rotate_accounts::verify(
        accounts,
        target_accounts_name,
        target_accounts_len,
        tx_accounts,
    )
}

// [`verify()`] does not have access to a valid bank, and so it can not call
// [`Bank::get_transaction_account_lock_limit()`].  This function does this check again, but during
// the generator execution, panicking if the limit was indeed exceeded.
fn verify_tx_max_accounts_during_generation(
    bank: &Bank,
    target_accounts_name: &'static str,
    tx_accounts: usize,
) {
    let account_lock_limit = bank.get_transaction_account_lock_limit();

    if tx_accounts <= account_lock_limit {
        return;
    }

    if account_lock_limit >= MAX_TX_ACCOUNT_LOCKS {
        panic!("`tx_max_accounts` passed into `generator()` exceeds one passed into `verify()`");
    } else {
        panic!(
            "`bank.get_transaction_account_lock_limit()` is above \
             `MAX_TX_ACCOUNT_LOCKS`.\nbank.get_transaction_account_lock_limit(): \
             {account_lock_limit}\nMAX_TX_ACCOUNT_LOCKS: {MAX_TX_ACCOUNT_LOCKS}\n`verify()` was \
             not able to check that the number of `{target_accounts_name}` account is adequate.",
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
        let tx_accounts = TX_MAX_ATTACK_ACCOUNTS_IN_PACKET.min(
            // The very first account will be our payer, while the rest can be populated by our
            // payload.
            bank.get_transaction_account_lock_limit().saturating_sub(1),
        );

        verify_tx_max_accounts_during_generation(bank, target_accounts_name, tx_accounts);

        rotate(tx_accounts, bank)
    }
}

#[cfg(test)]
mod tests {
    use {
        super::TX_PADDING, bincode, solana_hash::Hash,
        solana_message::legacy::Message as LegacyMessage, solana_pubkey::Pubkey,
    };

    #[test]
    fn check_tx_padding() {
        let payer = Pubkey::new_unique();

        let mut message = LegacyMessage::new(&[], Some(&payer));
        message.recent_blockhash = Hash::new_unique();

        let bytes = bincode::serialize(&message).unwrap();

        assert_eq!(
            bytes.len(),
            TX_PADDING,
            "`TX_PADDING` should be set to the size of an empty message"
        );
    }
}
