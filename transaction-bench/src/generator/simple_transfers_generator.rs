use {
    crate::{
        cli::SimpleTransferTxParams,
        generator::{
            transaction_batch_utils::spawn_blocking_transaction_batch_generation,
            transaction_builder::create_serialized_signed_transaction,
        },
    },
    rand::{seq::IteratorRandom, thread_rng},
    solana_hash::Hash,
    solana_keypair::Keypair,
    solana_signer::Signer,
    solana_system_interface::instruction as system_instruction,
    std::sync::Arc,
    tokio::task::JoinHandle,
};

// Generates a transaction batch of simple lamport transfer transactions.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn generate_transfer_transaction_batch(
    payers: Arc<Vec<Keypair>>,
    mut payer_index: usize,
    blockhash: Hash,
    params: SimpleTransferTxParams,
    send_batch_size: usize,
) -> JoinHandle<Vec<Vec<u8>>> {
    spawn_blocking_transaction_batch_generation("generate transfer transaction batch", move || {
        let mut txs: Vec<Vec<u8>> = Vec::with_capacity(send_batch_size);
        let lamports_to_transfer =
            unique_random_numbers(send_batch_size, params.lamports_to_transfer);
        for lamports in lamports_to_transfer.into_iter() {
            let payer = &payers[payer_index];
            payer_index = (payer_index + 1) % payers.len();

            let receiver = &payers[payer_index];
            payer_index = (payer_index + 1) % payers.len();

            let tx = create_serialized_signed_transaction(
                payer,
                blockhash,
                vec![system_instruction::transfer(
                    &payer.pubkey(),
                    &receiver.pubkey(),
                    lamports,
                )],
                vec![],
                params.transfer_tx_cu_budget,
            );

            txs.push(tx);
        }
        txs
    })
}

fn unique_random_numbers(count: usize, lamports_to_transfer: u64) -> Vec<u64> {
    assert!(
        count as u64 <= lamports_to_transfer,
        "Not enough unique values in range"
    );

    let mut rng = thread_rng();

    // Sample `count` unique values from the full range
    (1..=lamports_to_transfer).choose_multiple(&mut rng, count)
}
