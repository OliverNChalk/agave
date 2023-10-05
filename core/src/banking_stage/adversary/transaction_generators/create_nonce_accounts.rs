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

pub(super) fn generator(accounts: Arc<AccountsFile>, num_workers: usize) -> TransactionGenerator {
    let mut worker_index = 0;
    Box::new(move |bank: &Bank| {
        const BATCH_SIZE: usize = TARGET_NUM_TRANSACTIONS_PER_BATCH;
        let balance = Rent::default().minimum_balance(nonce::state::State::size());

        let mut transactions = vec![];
        let mut payers = accounts
            .payers
            .choose_multiple(&mut thread_rng(), BATCH_SIZE);
        for _ in 0..BATCH_SIZE {
            let payer = payers.next().unwrap();
            let nonce_account = Keypair::new();
            let instr = system_instruction::create_nonce_account(
                &payer.pubkey(),
                &nonce_account.pubkey(),
                &payer.pubkey(), // Make the fee payer the nonce account authority
                balance,
            );
            let mut tx_create_nonce_account =
                Transaction::new_with_payer(&instr, Some(&payer.pubkey()));

            tx_create_nonce_account
                .try_sign(&[&nonce_account, payer], bank.last_blockhash())
                .unwrap();

            transactions.push(SanitizedTransaction::from_transaction_for_tests(
                tx_create_nonce_account,
            ));
        }

        let current_worker_index = worker_index;
        worker_index = (worker_index + 1) % num_workers;
        (transactions, current_worker_index)
    })
}
