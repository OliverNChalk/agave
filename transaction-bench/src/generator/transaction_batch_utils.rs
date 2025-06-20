use {
    log::{debug, error},
    solana_measure::measure::Measure,
    solana_tpu_client_next::transaction_batch::TransactionBatch,
    tokio::{sync::mpsc::Sender, task::JoinHandle},
};

/// Helper to spawn a blocking task for generating a batch of transactions.
/// Manages performance measurement and logging.
pub(crate) fn spawn_blocking_transaction_batch_generation<F>(
    batch_description: &'static str,
    generation_logic: F,
) -> JoinHandle<Vec<Vec<u8>>>
where
    F: FnOnce() -> Vec<Vec<u8>> + Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        let mut measure_generate = Measure::start(batch_description);
        let txs = generation_logic();
        measure_generate.stop();
        debug!(
            "Time to {}: {} us, num transactions in batch: {}",
            batch_description,
            measure_generate.as_us(),
            txs.len(),
        );
        txs
    })
}

pub(crate) async fn send_batch(
    wired_txs_batch: Vec<Vec<u8>>,
    transactions_sender: Sender<TransactionBatch>,
) {
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
