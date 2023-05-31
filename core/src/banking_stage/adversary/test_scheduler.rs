//! Simple scheduler that drops network packets and generates transactions.
//!     - this is useful for testing the banking stage without the network
//!       or in creating stress-tests on a local network.

use {
    super::test_generators::TransactionGenerator,
    crate::{
        banking_stage::{
            committer::Committer,
            consumer::Consumer,
            decision_maker::DecisionMaker,
            qos_service::QosService,
            scheduler_messages::{ConsumeWork, FinishedConsumeWork, MaxAge},
            transaction_scheduler::batch_id_generator::BatchIdGenerator,
            BankingStage, BankingStageContext, ConsumeWorker,
        },
        validator::{BlockGeneratorConfig, BlockProductionMethod},
    },
    agave_banking_stage_ingress_types::BankingPacketReceiver,
    crossbeam_channel::{unbounded, Receiver, Sender, TryRecvError},
    solana_ledger::blockstore_processor::TransactionStatusSender,
    solana_poh::{poh_recorder::PohRecorder, transaction_recorder::TransactionRecorder},
    solana_runtime::{
        bank_forks::BankForks, prioritization_fee_cache::PrioritizationFeeCache,
        vote_sender_types::ReplayVoteSender,
    },
    solana_runtime_transaction::runtime_transaction::RuntimeTransaction,
    solana_transaction::sanitized::SanitizedTransaction,
    std::{
        num::NonZeroUsize,
        ops::Deref as _,
        sync::{atomic::AtomicBool, Arc, RwLock},
        thread::{sleep, Builder, JoinHandle},
        time::Duration,
    },
};

// An adversarial banking stage, used for testing replay.
// NOTE This is mostly copied from `BankingStage::new_num_threads()`.
#[allow(clippy::too_many_arguments)]
pub fn new_adverserial_banking_stage(
    block_production_method: BlockProductionMethod,
    poh_recorder: &Arc<RwLock<PohRecorder>>,
    transaction_recorder: TransactionRecorder,
    non_vote_receiver: BankingPacketReceiver,
    tpu_vote_receiver: BankingPacketReceiver,
    gossip_vote_receiver: BankingPacketReceiver,
    block_production_num_workers: NonZeroUsize,
    transaction_status_sender: Option<TransactionStatusSender>,
    replay_vote_sender: ReplayVoteSender,
    log_messages_bytes_limit: Option<usize>,
    bank_forks: Arc<RwLock<BankForks>>,
    prioritization_fee_cache: &Arc<PrioritizationFeeCache>,
    block_generator_config: BlockGeneratorConfig,
) -> BankingStage {
    match block_production_method {
        BlockProductionMethod::CentralScheduler | BlockProductionMethod::CentralSchedulerGreedy => {
            // `banking_stage::new_num_threads()` distinguishes between two scheduler types.  But we
            // do not care, and so we do not need the `use_greedy_scheduler` flag.

            new_adverserial_central_scheduler(
                poh_recorder,
                transaction_recorder,
                non_vote_receiver,
                tpu_vote_receiver,
                gossip_vote_receiver,
                block_production_num_workers,
                transaction_status_sender,
                replay_vote_sender,
                log_messages_bytes_limit,
                bank_forks,
                prioritization_fee_cache,
                block_generator_config,
            )
        }
    }
}

// NOTE This is mostly copied from `BankingStage::new_central_scheduler()`.
#[allow(clippy::too_many_arguments)]
fn new_adverserial_central_scheduler(
    poh_recorder: &Arc<RwLock<PohRecorder>>,
    transaction_recorder: TransactionRecorder,
    non_vote_receiver: BankingPacketReceiver,
    tpu_vote_receiver: BankingPacketReceiver,
    gossip_vote_receiver: BankingPacketReceiver,
    num_workers: NonZeroUsize,
    transaction_status_sender: Option<TransactionStatusSender>,
    replay_vote_sender: ReplayVoteSender,
    log_messages_bytes_limit: Option<usize>,
    bank_forks: Arc<RwLock<BankForks>>,
    prioritization_fee_cache: &Arc<PrioritizationFeeCache>,
    block_generator_config: BlockGeneratorConfig,
) -> BankingStage {
    assert!(num_workers <= BankingStage::max_num_workers());

    let decision_maker = DecisionMaker::from(poh_recorder.read().unwrap().deref());
    let committer = Committer::new(
        transaction_status_sender.clone(),
        replay_vote_sender.clone(),
        prioritization_fee_cache.clone(),
    );

    let context = BankingStageContext {
        exit_signal: Arc::new(AtomicBool::new(false)),
        tpu_vote_receiver,
        gossip_vote_receiver,
        non_vote_receiver,
        transaction_recorder,
        poh_recorder: poh_recorder.clone(),
        bank_forks,
        committer,
        log_messages_bytes_limit,
    };

    // + 1 for vote worker
    // + 1 for the scheduler thread
    let mut thread_hdls = Vec::with_capacity(num_workers.get() + 2);
    thread_hdls.push(BankingStage::spawn_vote_worker(&context));

    // banking_stage::new_num_threads() is parametrized over the `transaction_struct`, allowing it
    // to accept either `SanitizedTransaction` or `ResolvedTransactionView`.  But the adversarial
    // scheduler only produces the `SanitizedTransaction`, so making this code generic over both is
    // more work than necessary.

    let (_, thread_hdls) = spawn_adverserial_scheduler_and_workers(
        context.non_vote_receiver.clone(),
        decision_maker,
        context.committer.clone(),
        poh_recorder,
        context.transaction_recorder.clone(),
        num_workers,
        log_messages_bytes_limit,
        block_generator_config,
    );

    BankingStage {
        context: Some(context),
        thread_hdls,
    }
}

// NOTE This is mostly copied from `BankingStage::spawn_scheduler_and_workers()`.
#[allow(clippy::too_many_arguments)]
fn spawn_adverserial_scheduler_and_workers(
    non_vote_receiver: BankingPacketReceiver,
    decision_maker: DecisionMaker,
    committer: Committer,
    poh_recorder: &Arc<RwLock<PohRecorder>>,
    transaction_recorder: TransactionRecorder,
    num_workers: NonZeroUsize,
    log_messages_bytes_limit: Option<usize>,
    block_generator_config: BlockGeneratorConfig,
) -> (Arc<AtomicBool>, Vec<JoinHandle<()>>) {
    let num_workers = num_workers.get();

    // Create channels for communication between scheduler and workers
    let (work_senders, work_receivers): (Vec<Sender<_>>, Vec<Receiver<_>>) =
        (0..num_workers).map(|_| unbounded()).unzip();
    let (finished_work_sender, finished_work_receiver) = unbounded();

    let exit = Arc::new(AtomicBool::new(false));
    // + 1 for the central scheduler thread
    let mut non_vote_thread_hdls = Vec::with_capacity(num_workers + 1);

    // Spawn the worker threads
    let mut worker_metrics = Vec::with_capacity(num_workers);
    for (index, work_receiver) in work_receivers.into_iter().enumerate() {
        let id = index as u32;
        let consume_worker = ConsumeWorker::new(
            id,
            exit.clone(),
            work_receiver,
            Consumer::new(
                committer.clone(),
                transaction_recorder.clone(),
                QosService::new(id),
                log_messages_bytes_limit,
            ),
            finished_work_sender.clone(),
            poh_recorder.read().unwrap().shared_leader_state(),
        );

        worker_metrics.push(consume_worker.metrics_handle());
        non_vote_thread_hdls.push(
            Builder::new()
                .name(format!("solCoWorker{id:02}"))
                .spawn(move || {
                    let _ = consume_worker.run();
                })
                .unwrap(),
        )
    }

    // Scheduler thread
    let scheduler = TestScheduler::new(
        decision_maker,
        non_vote_receiver,
        work_senders,
        finished_work_receiver,
        crate::banking_stage::adversary::test_generators::get_transaction_generators(
            &block_generator_config,
            num_workers,
        ),
    );
    non_vote_thread_hdls.push(
        Builder::new()
            .name("solAdvBnkTxSched".to_string())
            .spawn(move || scheduler.run())
            .unwrap(),
    );

    (exit, non_vote_thread_hdls)
}

/// This is an adversarial scheduler.  It does not process any incoming transactions, but instead
/// will generate transactions using our test generators.
///
// It is similar to `banking_stage::transaction_scheduler::greedy_scheduler::GreedyScheduler`.
// Mostly the `SchedulingCommon` part.
pub struct TestScheduler {
    /// Decision maker - only generate when leader
    decision_maker: DecisionMaker,

    /// From SigVerify - ignored
    non_vote_receiver: BankingPacketReceiver,

    /// To BankingStageWorker(s)
    consume_work_senders: Vec<Sender<ConsumeWork<RuntimeTransaction<SanitizedTransaction>>>>,

    /// From BankingStageWorker
    finished_consume_work_receiver:
        Receiver<FinishedConsumeWork<RuntimeTransaction<SanitizedTransaction>>>,

    /// Transaction batch generators
    transaction_generators: Vec<(TransactionGenerator, usize)>,

    /// Index of the transaction generator to use
    tx_gen_idx: usize,
}

impl TestScheduler {
    pub fn new(
        decision_maker: DecisionMaker,
        non_vote_receiver: BankingPacketReceiver,
        consume_work_senders: Vec<Sender<ConsumeWork<RuntimeTransaction<SanitizedTransaction>>>>,
        finished_consume_work_receiver: Receiver<
            FinishedConsumeWork<RuntimeTransaction<SanitizedTransaction>>,
        >,
        transaction_generators: Vec<(TransactionGenerator, usize)>,
    ) -> Self {
        Self {
            decision_maker,
            non_vote_receiver,
            consume_work_senders,
            finished_consume_work_receiver,
            transaction_generators,
            tx_gen_idx: 0,
        }
    }

    fn advance_tx_gen_idx(&mut self) {
        self.tx_gen_idx = (self.tx_gen_idx + 1) % self.transaction_generators.len();
    }

    pub fn run(mut self) {
        let mut bank_slot = 0;
        let mut tx_batch_id_gen = BatchIdGenerator::default();
        let mut tx_id: usize = 0;

        'scheduler_loop: loop {
            loop {
                match self.non_vote_receiver.try_recv() {
                    Ok(_packet_batch) => {
                        // Drop incoming packets from sigverify so we don't OOM
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        debug!("TestScheduler exiting");
                        break 'scheduler_loop;
                    }
                }
            }

            // Drop finished work from banking stage so we don't OOM
            self.finished_consume_work_receiver
                .try_iter()
                .for_each(drop);

            let decision = self.decision_maker.make_consume_or_forward_decision();

            let Some(bank) = decision.bank() else {
                // If we do not have a leader slot, then we are not the leader and we are not going
                // to process any traffic.  Wait a bit to avoid blocking the CPU.
                sleep(Duration::from_millis(1));
                continue;
            };

            if bank_slot != bank.slot() {
                // Switch to next transaction generator on slot boundary
                bank_slot = bank.slot();
                self.advance_tx_gen_idx();
                trace!(
                    "Generating transactions for slot {bank_slot} w/ generator {}",
                    self.tx_gen_idx
                );
            }

            let (generator, num_tx_batches) = &mut self.transaction_generators[self.tx_gen_idx];
            // Batch transactions to amortize decision cost.
            for _ in 0..*num_tx_batches {
                let (transactions, worker_index) = (*generator)(bank);

                let transactions = transactions
                    .into_iter()
                    .map(|tx| {
                        RuntimeTransaction::new_from_sanitized(tx).unwrap_or_else(|err| {
                            panic!(
                                "Failed to convert a `SanitizedTransaction` produced by the \
                                 generator into a \
                                 `RuntimeTransaction<SanitizedTransaction>`.\nError: {err}",
                            )
                        })
                    })
                    .collect::<Vec<_>>();

                let tx_count = transactions.len();
                let scheduled_work = ConsumeWork::<RuntimeTransaction<SanitizedTransaction>> {
                    batch_id: tx_batch_id_gen.next(),
                    ids: (0..tx_count)
                        .map(|_| {
                            tx_id += 1;
                            tx_id
                        })
                        .collect(),
                    transactions,
                    max_ages: (0..tx_count).map(|_| MaxAge::MAX).collect(),
                };

                self.consume_work_senders[worker_index]
                    .send(scheduled_work)
                    .unwrap();
            }
        }
    }
}
