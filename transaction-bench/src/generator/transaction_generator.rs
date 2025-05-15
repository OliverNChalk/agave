//! Service generating serialized transactions in batches.
use {
    crate::{
        accounts_file::AccountsFile,
        cli::{ReadAccountsTxParams, SimpleTransferTxParams, TransactionParams, WorkloadParams},
        generator::{
            chunked_accounts_iterator::ChunkedAccountsIterator,
            transaction_type::{generate_tx_type_sequence, TransactionType},
        },
    },
    client_test_program::ClientTestProgramInstruction,
    log::*,
    rand::{seq::IteratorRandom, thread_rng, RngCore},
    solana_compute_budget_interface::ComputeBudgetInstruction,
    solana_hash::Hash,
    solana_instruction::{AccountMeta, Instruction},
    solana_keypair::Keypair,
    solana_measure::measure::Measure,
    solana_message::Message,
    solana_pubkey::Pubkey,
    solana_signer::Signer,
    solana_system_interface::instruction as system_instruction,
    solana_tpu_client_next::transaction_batch::TransactionBatch,
    solana_transaction::Transaction,
    std::sync::Arc,
    thiserror::Error,
    tokio::{
        sync::{mpsc::Sender, watch},
        task::{JoinHandle, JoinSet},
        time::{Duration, Instant},
    },
};

#[derive(Error, Debug)]
pub enum TransactionGeneratorError {
    #[error("Transactions receiver has been dropped unexpectedly.")]
    ReceiverDropped,

    #[error("Failed to generate transaction batch.")]
    GenerateTxBatchFailure,
}

pub struct TransactionGenerator {
    accounts: AccountsFile,
    blockhash_receiver: watch::Receiver<Hash>,
    transactions_sender: Sender<TransactionBatch>,
    transaction_params: TransactionParams,
    workload_params: WorkloadParams,
    send_batch_size: usize,
    run_duration: Option<Duration>,
    workers_pull_size: usize,
}

impl TransactionGenerator {
    pub fn new(
        accounts: AccountsFile,
        blockhash_receiver: watch::Receiver<Hash>,
        transactions_sender: Sender<TransactionBatch>,
        transaction_params: TransactionParams,
        workload_params: WorkloadParams,
        send_batch_size: usize,
        duration: Option<Duration>,
        workers_pull_size: usize,
    ) -> Self {
        Self {
            accounts,
            blockhash_receiver,
            transactions_sender,
            transaction_params,
            workload_params,
            send_batch_size,
            run_duration: duration,
            workers_pull_size,
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
    pub async fn run(self) -> Result<(), TransactionGeneratorError> {
        let accounts_meta: Arc<Vec<AccountMeta>> = Arc::new(
            self.accounts
                .sized_accounts
                .iter()
                .map(|account| AccountMeta {
                    pubkey: account.pubkey(),
                    is_signer: false,
                    is_writable: false,
                })
                .collect(),
        );
        let len_accounts_meta = accounts_meta.len();
        let payers = Arc::new(self.accounts.payers);
        let len_payers = payers.len();
        let mut index_payer: usize = 0;
        let mut accounts_begin: usize = 0;
        let mut futures = JoinSet::new();

        let tx_type_sequence = generate_tx_type_sequence(
            self.workload_params.transaction_mix.read_accounts_pct,
            self.workload_params.transaction_mix.simple_transfer_pct,
        );
        debug!("Transaction type sequence: {tx_type_sequence:?}");
        let mut tx_type_iter = tx_type_sequence.iter().copied().cycle();

        let start = Instant::now();
        loop {
            if let Some(run_duration) = self.run_duration {
                if start.elapsed() >= run_duration {
                    info!("Transaction generator is stopping...");
                    while let Some(result) = futures.join_next().await {
                        debug!("Future result {result:?}");
                    }
                    break;
                }
            }
            if self.transactions_sender.is_closed() {
                return Err(TransactionGeneratorError::ReceiverDropped);
            }
            let blockhash = *self.blockhash_receiver.borrow();

            while futures.len() < self.workers_pull_size {
                let send_batch_size = self.send_batch_size;
                let transaction_params = self.transaction_params.clone();
                let accounts_meta = accounts_meta.clone();
                let payers = payers.clone();
                let owner_program_id = self.accounts.owner_program_id;
                let transactions_sender = self.transactions_sender.clone();

                let transaction_type = tx_type_iter.next().unwrap();

                match transaction_type {
                    TransactionType::Read => {
                        let num_accounts_in_tx: Vec<usize> = (0..send_batch_size)
                            .map(|_| {
                                self.transaction_params
                                    .read_accounts_tx_params
                                    .num_accounts_per_tx
                                    .uniform()
                            })
                            .collect();
                        let num_batch_accounts: usize = num_accounts_in_tx.iter().sum();
                        let accounts_end = accounts_begin.saturating_add(num_batch_accounts);

                        futures.spawn(async move {
                            let Ok(wired_tx_batch) =
                                Self::generate_read_accounts_transaction_batch(
                                    accounts_meta,
                                    accounts_begin,
                                    payers,
                                    index_payer,
                                    blockhash,
                                    transaction_params.read_accounts_tx_params,
                                    owner_program_id,
                                    send_batch_size,
                                    num_accounts_in_tx,
                                )
                                .await
                            else {
                                warn!("Failed to generate txs batch!");
                                return;
                            };

                            send_batch(wired_tx_batch, transactions_sender).await;
                        });
                        index_payer = index_payer.saturating_add(self.send_batch_size) % len_payers;
                        accounts_begin = accounts_end % len_accounts_meta;
                    }
                    TransactionType::Transfer => {
                        futures.spawn(async move {
                            let Ok(wired_tx_batch) = Self::generate_transfer_transaction_batch(
                                payers,
                                index_payer,
                                blockhash,
                                transaction_params.simple_transfer_tx_params,
                                send_batch_size,
                            )
                            .await
                            else {
                                warn!("Failed to generate txs batch!");
                                return;
                            };

                            send_batch(wired_tx_batch, transactions_sender).await;
                        });
                        // 2 * self.send_batch_size because for simple transfer
                        // we form transactions as follows: p1 -> p2, p3 -> p4,
                        // etc.
                        // Note, that sized accounts are not used so there is no reason to increment this counter.
                        index_payer =
                            index_payer.saturating_add(2 * self.send_batch_size) % len_payers;
                    }
                }
            }
            futures.join_next().await;
        }
        Ok(())
    }

    /// Generate transaction batch in a spawn_blocking task.
    /// We need to spawn_blocking because signing and serializing transactions
    /// is computationally expensive (~26us per tx).
    #[allow(clippy::arithmetic_side_effects)]
    fn generate_read_accounts_transaction_batch(
        accounts_meta: Arc<Vec<AccountMeta>>,
        accounts_begin: usize,
        payers: Arc<Vec<Keypair>>,
        mut payer_index: usize,
        blockhash: Hash,
        transaction_params: ReadAccountsTxParams,
        program_id: Pubkey,
        send_batch_size: usize,
        num_accounts_per_tx: Vec<usize>,
    ) -> JoinHandle<Vec<Vec<u8>>> {
        tokio::task::spawn_blocking(move || {
            let mut measure_generate = Measure::start("generate transaction batch");
            let accounts_chunk_it =
                ChunkedAccountsIterator::new(accounts_meta, accounts_begin, &num_accounts_per_tx);
            let txs: Vec<Vec<u8>> = accounts_chunk_it
                .map(|tx_accounts| {
                    let payer = &payers[payer_index];
                    payer_index = (payer_index + 1) % payers.len();
                    let message = create_read_message(
                        payer,
                        &program_id,
                        tx_accounts,
                        transaction_params.read_tx_cu_budget,
                    );
                    let tx = Transaction::new(&[payer], message, blockhash);
                    bincode::serialize(&tx).expect("serialize Transaction in send_batch")
                })
                .collect();
            measure_generate.stop();
            debug!(
                "Time to generate transaction batch: {} us, num transactions in batches: {}",
                measure_generate.as_us(),
                send_batch_size
            );
            txs
        })
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn generate_transfer_transaction_batch(
        payers: Arc<Vec<Keypair>>,
        mut payer_index: usize,
        blockhash: Hash,
        params: SimpleTransferTxParams,
        send_batch_size: usize,
    ) -> JoinHandle<Vec<Vec<u8>>> {
        tokio::task::spawn_blocking(move || {
            let mut measure_generate = Measure::start("generate transfer transaction batch");
            let mut txs: Vec<Vec<u8>> = Vec::with_capacity(send_batch_size);
            let lamports_to_transfer =
                unique_random_numbers(send_batch_size, params.lamports_to_transfer);
            for lamports in lamports_to_transfer.into_iter() {
                let payer = &payers[payer_index];
                payer_index = (payer_index + 1) % payers.len();

                let receiver = &payers[payer_index];
                payer_index = (payer_index + 1) % payers.len();

                let message = create_transfer_message(
                    payer,
                    &receiver.pubkey(),
                    params.transfer_tx_cu_budget,
                    lamports,
                );
                let tx = Transaction::new(&[payer], message, blockhash);

                txs.push(bincode::serialize(&tx).expect("serialize Transaction in send_batch"));
            }
            measure_generate.stop();
            debug!(
                "Time to generate transfer transaction batch: {} us, num transactions in batches: \
                 {}",
                measure_generate.as_us(),
                send_batch_size
            );
            txs
        })
    }
}

fn create_read_message(
    payer: &Keypair,
    program_id: &Pubkey,
    accounts_meta: Vec<AccountMeta>,
    transaction_cu_budget: u32,
) -> Message {
    let data = ClientTestProgramInstruction::ReadAccounts {
        random: thread_rng().next_u64(),
    };

    // Explicitly specify the CU budget to avoid dropping some transactions on the CU check side.
    // The constraint is that 48M/64 > CU limit.
    // To maximize number of txs in the block, it is beneficial to set CU limit to be close to the
    // real CU consumption.
    // Default is 200k.
    let set_cu_instruction =
        ComputeBudgetInstruction::set_compute_unit_limit(transaction_cu_budget);

    let read_instruction = Instruction::new_with_borsh(*program_id, &data, accounts_meta);
    Message::new(
        &[set_cu_instruction, read_instruction],
        Some(&payer.pubkey()),
    )
}

fn create_transfer_message(
    payer: &Keypair,
    receiver: &Pubkey,
    transfer_tx_cu_budget: u32,
    lamports_to_transfer: u64,
) -> Message {
    let set_cu_instruction =
        ComputeBudgetInstruction::set_compute_unit_limit(transfer_tx_cu_budget);
    // We don't set_loaded_accounts_data_size_limit because it costs 150 CU but doesn't save any.

    let transfer_instruction =
        system_instruction::transfer(&payer.pubkey(), receiver, lamports_to_transfer);
    Message::new(
        &[set_cu_instruction, transfer_instruction],
        Some(&payer.pubkey()),
    )
}

async fn send_batch(wired_txs_batch: Vec<Vec<u8>>, transactions_sender: Sender<TransactionBatch>) {
    let mut measure_send_to_queue = Measure::start("add transaction batch to channel");
    if let Err(err) = transactions_sender
        .send(TransactionBatch::new(wired_txs_batch))
        .await
    {
        error!("Receiver dropped, error {err}.");
        return;
    }
    measure_send_to_queue.stop();
    debug!(
        "Time to send into transactions queue: {} us",
        measure_send_to_queue.as_us()
    );
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
