//! Generates nonce accounts.

use {
    super::TransactionGenerator,
    crate::banking_stage::consumer::TARGET_NUM_TRANSACTIONS_PER_BATCH,
    rand::{seq::SliceRandom, thread_rng},
    solana_adversary::accounts_file::AccountsFile,
    solana_keypair::Keypair,
    solana_nonce as nonce,
    solana_rent::Rent,
    solana_runtime::bank::Bank,
    solana_signer::Signer,
    solana_system_interface::instruction as system_instruction,
    solana_transaction::{sanitized::SanitizedTransaction, Transaction},
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
    let balance = Rent::default().minimum_balance(nonce::state::State::size());

    let mut worker_index = 0;
    Box::new(move |bank: &Bank| {
        let blockhash = bank.last_blockhash();

        let transactions = accounts
            .payers
            .choose_multiple(&mut thread_rng(), BATCH_SIZE)
            .map(|payer| {
                let nonce_account = Keypair::new();
                let instr = system_instruction::create_nonce_account(
                    &payer.pubkey(),
                    &nonce_account.pubkey(),
                    // Make the fee payer the nonce account authority
                    &payer.pubkey(),
                    balance,
                );

                Transaction::new_signed_with_payer(
                    &instr,
                    Some(&payer.pubkey()),
                    &[&nonce_account, payer],
                    blockhash,
                )
            })
            .map(SanitizedTransaction::from_transaction_for_tests)
            .collect::<Vec<_>>();

        let current_worker_index = worker_index;
        worker_index = (worker_index + 1) % num_workers;
        (transactions, current_worker_index)
    })
}
