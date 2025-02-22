//! Service generating serialized transactions in batches.
use {
    crate::{
        accounts_file::AccountsFile,
        cli::{ReadAccountsTxParams, SimpleTransferTxParams, TransactionParams, WorkloadParams},
        generator::chunked_accounts_iterator::ChunkedAccountsIterator,
    },
    client_test_program::ClientTestProgramInstruction,
    log::*,
    rand::{thread_rng, RngCore},
    solana_compute_budget_interface::ComputeBudgetInstruction,
    solana_hash::Hash,
    solana_instruction::{AccountMeta, Instruction},
    solana_keypair::Keypair,
    solana_measure::measure::Measure,
    solana_message::Message,
    solana_pubkey::Pubkey,
    solana_signer::Signer,
    solana_system_transaction as system_transaction,
    solana_tpu_client_next::transaction_batch::TransactionBatch,
    solana_transaction::Transaction,
    std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thiserror::Error,
    tokio::{
        sync::{mpsc::Sender, watch},
        task::{JoinHandle, JoinSet},
        time::{Duration, Instant},
    },
};

/// The max size of the JoinSet container used to execute futures concurrently and in parallel.
/// It should be well-tuned because we don't want to generate more transactions than we can send.
const MAX_JOIN_SET_SIZE: usize = 4;

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
    ) -> Self {
        Self {
            accounts,
            blockhash_receiver,
            transactions_sender,
            transaction_params,
            workload_params,
            send_batch_size,
            run_duration: duration,
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
        let read_accounts_count = Arc::new(AtomicUsize::new(0));
        let transfer_count = Arc::new(AtomicUsize::new(0));
        let read_accounts_limit = self.workload_params.transaction_mix.read_accounts_pct;
        let transfer_limit = self.workload_params.transaction_mix.simple_transfer_pct;

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

            while futures.len() < MAX_JOIN_SET_SIZE {
                let send_batch_size = self.send_batch_size;
                let transaction_params = self.transaction_params.clone();
                let accounts_meta = accounts_meta.clone();
                let payers = payers.clone();
                let owner_program_id = self.accounts.owner_program_id;
                let transactions_sender = self.transactions_sender.clone();
                let read_accounts_count = read_accounts_count.clone();
                let transfer_count = transfer_count.clone();

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
                    let wire_tx_batch = Self::generate_transaction_batch(
                        read_accounts_count,
                        read_accounts_limit,
                        transfer_count,
                        transfer_limit,
                        accounts_meta,
                        accounts_begin,
                        payers,
                        index_payer,
                        blockhash,
                        transaction_params,
                        owner_program_id,
                        send_batch_size,
                        num_accounts_in_tx,
                    )
                    .await;
                    let Ok(wire_txs_batch) = wire_tx_batch else {
                        warn!("Failed to generate txs batch!");
                        return;
                    };

                    let mut measure_send_to_queue =
                        Measure::start("add transaction batch to channel");
                    if let Err(err) = transactions_sender
                        .send(TransactionBatch::new(wire_txs_batch))
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
                });
                index_payer = index_payer.saturating_add(self.send_batch_size) % len_payers;
                accounts_begin = accounts_end % len_accounts_meta;
            }
            futures.join_next().await;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_transaction_batch(
        read_accounts_count: Arc<AtomicUsize>,
        read_accounts_limit: usize,
        transfer_count: Arc<AtomicUsize>,
        transfer_limit: usize,
        accounts_meta: Arc<Vec<AccountMeta>>,
        accounts_begin: usize,
        payers: Arc<Vec<Keypair>>,
        index_payer: usize,
        blockhash: Hash,
        transaction_params: TransactionParams,
        owner_program_id: Pubkey,
        send_batch_size: usize,
        num_accounts_in_tx: Vec<usize>,
    ) -> JoinHandle<Vec<Vec<u8>>> {
        let mut read_accounts_tx_limits_reached =
            read_accounts_count.load(Ordering::Relaxed) >= read_accounts_limit;
        let mut transfer_tx_limits_reached =
            transfer_count.load(Ordering::Relaxed) >= transfer_limit;

        // Start over when both limits are reached.
        if read_accounts_tx_limits_reached && transfer_tx_limits_reached {
            read_accounts_count.store(0, Ordering::Relaxed);
            transfer_count.store(0, Ordering::Relaxed);
            read_accounts_tx_limits_reached = read_accounts_limit == 0;
            transfer_tx_limits_reached = transfer_limit == 0;
        }

        if !read_accounts_tx_limits_reached {
            read_accounts_count.fetch_add(1, Ordering::Relaxed);
            Self::generate_read_accounts_transaction_batch(
                accounts_meta.clone(),
                accounts_begin,
                payers.clone(),
                index_payer,
                blockhash,
                transaction_params.read_accounts_tx_params,
                owner_program_id,
                send_batch_size,
                num_accounts_in_tx,
            )
        } else if !transfer_tx_limits_reached {
            transfer_count.fetch_add(1, Ordering::Relaxed);
            Self::generate_transfer_transaction_batch(
                payers.clone(),
                index_payer,
                blockhash,
                transaction_params.simple_transfer_tx_params,
                send_batch_size,
            )
        } else {
            panic!("All transaction types limits were reached and incorrectly not reset.");
        }
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
                        transaction_params.transaction_cu_budget,
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

    /// Generate transaction batch in a spawn_blocking task.
    /// We need to spawn_blocking because signing and serializing transactions
    /// is computationally expensive (~26us per tx).
    #[allow(clippy::arithmetic_side_effects)]
    fn generate_transfer_transaction_batch(
        payers: Arc<Vec<Keypair>>,
        mut payer_index: usize,
        blockhash: Hash,
        transaction_params: SimpleTransferTxParams,
        send_batch_size: usize,
    ) -> JoinHandle<Vec<Vec<u8>>> {
        tokio::task::spawn_blocking(move || {
            let mut measure_generate = Measure::start("generate transfer transaction batch");
            let mut txs: Vec<Vec<u8>> = Vec::with_capacity(send_batch_size);
            for _ in 0..send_batch_size {
                let payer = &payers[payer_index];
                payer_index = (payer_index + 1) % payers.len();

                let receiver = &payers[payer_index];
                payer_index = (payer_index + 1) % payers.len();

                let tx = system_transaction::transfer(
                    payer,
                    &receiver.pubkey(),
                    transaction_params.lamports_to_transfer,
                    blockhash,
                );
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
