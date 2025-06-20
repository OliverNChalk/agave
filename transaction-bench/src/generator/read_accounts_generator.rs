use {
    crate::{
        cli::ReadAccountsTxParams,
        generator::{
            chunked_accounts_iterator::ChunkedAccountsIterator,
            transaction_batch_utils::spawn_blocking_transaction_batch_generation,
            transaction_builder::create_serialized_signed_transaction,
        },
    },
    client_test_program::ClientTestProgramInstruction,
    rand::{thread_rng, RngCore},
    solana_hash::Hash,
    solana_instruction::{AccountMeta, Instruction},
    solana_keypair::Keypair,
    solana_pubkey::Pubkey,
    std::sync::Arc,
    tokio::task::JoinHandle,
};

/// Generate transaction batch in a spawn_blocking task.
/// We need to spawn_blocking because signing and serializing transactions
/// is computationally expensive (~26us per tx).
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn generate_read_accounts_transaction_batch(
    accounts_meta: Arc<Vec<AccountMeta>>,
    accounts_begin: usize,
    payers: Arc<Vec<Keypair>>,
    mut payer_index: usize,
    blockhash: Hash,
    transaction_params: ReadAccountsTxParams,
    program_id: Pubkey,
    num_accounts_per_tx: Vec<usize>,
) -> JoinHandle<Vec<Vec<u8>>> {
    spawn_blocking_transaction_batch_generation(
        "generate read_accounts transaction batch",
        move || {
            let accounts_chunk_it =
                ChunkedAccountsIterator::new(accounts_meta, accounts_begin, &num_accounts_per_tx);
            let txs: Vec<Vec<u8>> = accounts_chunk_it
                .map(|tx_accounts| {
                    let payer = &payers[payer_index];
                    payer_index = (payer_index + 1) % payers.len();

                    let ix_data = ClientTestProgramInstruction::ReadAccounts {
                        random: thread_rng().next_u64(),
                    };
                    create_serialized_signed_transaction(
                        payer,
                        blockhash,
                        vec![Instruction::new_with_borsh(
                            program_id,
                            &ix_data,
                            tx_accounts,
                        )],
                        vec![],
                        transaction_params.read_tx_cu_budget,
                    )
                })
                .collect();
            txs
        },
    )
}
