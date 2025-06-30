//! Service generating serialized transactions in batches.
use {
    crate::{
        accounts_file::AccountsFile,
        cli::{TransactionParams, WorkloadParams},
        generator::{
            mints_generator::generate_mint_batch,
            read_accounts_generator::generate_read_accounts_transaction_batch,
            simple_transfers_generator::generate_transfer_transaction_batch,
            transaction_batch_utils::send_batch,
            transaction_type::{generate_tx_type_sequence, TransactionType},
        },
    },
    log::*,
    solana_hash::Hash,
    solana_instruction::AccountMeta,
    solana_signer::Signer,
    solana_tpu_client_next::transaction_batch::TransactionBatch,
    std::sync::Arc,
    thiserror::Error,
    tokio::{
        sync::{mpsc::Sender, watch},
        task::JoinSet,
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
            self.workload_params.transaction_mix.mint_pct,
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
                            let Ok(wired_tx_batch) = generate_read_accounts_transaction_batch(
                                accounts_meta,
                                accounts_begin,
                                payers,
                                index_payer,
                                blockhash,
                                transaction_params.read_accounts_tx_params,
                                owner_program_id,
                                num_accounts_in_tx,
                            )
                            .await
                            else {
                                warn!("Failed to generate readAccounts txs batch!");
                                return;
                            };

                            send_batch(wired_tx_batch, transactions_sender).await;
                        });
                        index_payer = index_payer.saturating_add(self.send_batch_size) % len_payers;
                        accounts_begin = accounts_end % len_accounts_meta;
                    }
                    TransactionType::Transfer => {
                        let num_send_instructions_per_tx = transaction_params
                            .simple_transfer_tx_params
                            .num_send_instructions_per_tx;
                        futures.spawn(async move {
                            let Ok(wired_tx_batch) = generate_transfer_transaction_batch(
                                payers,
                                index_payer,
                                blockhash,
                                transaction_params.simple_transfer_tx_params,
                                send_batch_size,
                            )
                            .await
                            else {
                                warn!("Failed to generate transfer txs batch!");
                                return;
                            };

                            send_batch(wired_tx_batch, transactions_sender).await;
                        });
                        // 2 * self.send_batch_size because for simple transfer
                        // we form transactions as follows: p1 -> p2, p3 -> p4,
                        // etc.
                        // Note, that sized accounts are not used so there is no reason to increment this counter.
                        index_payer = index_payer.saturating_add(
                            2 * num_send_instructions_per_tx * self.send_batch_size,
                        ) % len_payers;
                    }
                    TransactionType::Mint => {
                        futures.spawn(async move {
                            let Ok(wired_tx_batch) = generate_mint_batch(
                                payers,
                                index_payer,
                                blockhash,
                                transaction_params.mint_tx_params,
                                send_batch_size,
                            )
                            .await
                            else {
                                warn!("Failed to generate mint txs batch!");
                                return;
                            };

                            send_batch(wired_tx_batch, transactions_sender).await;
                        });
                        index_payer = index_payer.saturating_add(self.send_batch_size) % len_payers;
                    }
                }
            }
            futures.join_next().await;
        }
        Ok(())
    }
}
