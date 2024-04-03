use {
    crate::{
        banking_stage::{
            adversary::attack_scheduler::AttackScheduler, committer::Committer, consumer::Consumer,
            decision_maker::DecisionMaker, qos_service::QosService, BankingStage, ConsumeWorker,
        },
        validator::BlockProductionMethod,
    },
    crossbeam_channel::{bounded, unbounded, Receiver, Sender},
    solana_adversary::ReplayAttackReceiver,
    solana_ledger::blockstore_processor::TransactionStatusSender,
    solana_poh::{poh_recorder::PohRecorder, transaction_recorder::TransactionRecorder},
    solana_runtime::{
        prioritization_fee_cache::PrioritizationFeeCache, vote_sender_types::ReplayVoteSender,
    },
    std::{
        num::NonZeroUsize,
        ops::Deref as _,
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc, RwLock,
        },
        thread::{Builder, JoinHandle},
    },
};

// An adversarial banking stage, used for testing replay.
// NOTE This is mostly copied from `BankingStage`.
pub struct AdversarialBankingStage {
    non_vote_exit_signal: Arc<AtomicBool>,
    non_vote_thread_hdls: Vec<JoinHandle<()>>,
}

impl AdversarialBankingStage {
    /// Same functionality as in [`BankingStage::new()`], but adjusted for the needs of the
    /// invalidator.
    ///
    /// This bankings stage is designed to work alongside the normal [`BankingStage`] machinery.
    /// When an attack is active, and this node is the leader, the normal banking stage is disabled
    /// and this bankings stage generates transactions.  Otherwise, the `AdversarialBankingStage`
    /// does nothing and the normal [`BankingStage`] is processing all the transactions as expected.
    ///
    /// [`BankingStage::new()`]: crate::banking_stage::BankingStage::new()
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        block_production_method: BlockProductionMethod,
        poh_recorder: &Arc<RwLock<PohRecorder>>,
        transaction_recorder: TransactionRecorder,
        num_workers: NonZeroUsize,
        transaction_status_sender: Option<TransactionStatusSender>,
        replay_vote_sender: ReplayVoteSender,
        log_messages_bytes_limit: Option<usize>,
        prioritization_fee_cache: &Arc<PrioritizationFeeCache>,
        replay_attack_receiver: Option<ReplayAttackReceiver>,
        drop_packets: Arc<AtomicBool>,
    ) -> Self {
        match block_production_method {
            BlockProductionMethod::CentralScheduler
            | BlockProductionMethod::CentralSchedulerGreedy => {
                // `banking_stage::new_num_threads()` distinguishes between two scheduler types.
                // But we do not care, and so we do not need the `use_greedy_scheduler` flag.

                new_adversarial_central_scheduler(
                    poh_recorder,
                    transaction_recorder,
                    num_workers,
                    transaction_status_sender,
                    replay_vote_sender,
                    log_messages_bytes_limit,
                    prioritization_fee_cache,
                    replay_attack_receiver,
                    drop_packets,
                )
            }
        }
    }

    pub fn join(self) -> std::thread::Result<()> {
        self.non_vote_exit_signal.store(true, Ordering::Relaxed);
        for thread_hdl in self.non_vote_thread_hdls {
            thread_hdl.join()?;
        }
        Ok(())
    }
}

/// Functionality similar to [`BankingStage::new_central_scheduler()`], but adjusted to the needs of
/// the invalidator.
///
/// [`BankingStage::new_central_scheduler()`]: crate::banking_stage::BankingStage::new_central_scheduler()
#[allow(clippy::too_many_arguments)]
fn new_adversarial_central_scheduler(
    poh_recorder: &Arc<RwLock<PohRecorder>>,
    transaction_recorder: TransactionRecorder,
    num_workers: NonZeroUsize,
    transaction_status_sender: Option<TransactionStatusSender>,
    replay_vote_sender: ReplayVoteSender,
    log_messages_bytes_limit: Option<usize>,
    prioritization_fee_cache: &Arc<PrioritizationFeeCache>,
    replay_attack_receiver: Option<ReplayAttackReceiver>,
    drop_packets: Arc<AtomicBool>,
) -> AdversarialBankingStage {
    // If not configured for adversarial mode, spawn no threads.
    let Some(replay_attack_receiver) = replay_attack_receiver else {
        return AdversarialBankingStage {
            non_vote_exit_signal: Arc::new(AtomicBool::new(false)),
            non_vote_thread_hdls: vec![],
        };
    };

    assert!(num_workers <= BankingStage::max_num_workers());

    let decision_maker = DecisionMaker::from(poh_recorder.read().unwrap().deref());
    let committer = Committer::new(
        transaction_status_sender.clone(),
        replay_vote_sender.clone(),
        prioritization_fee_cache.clone(),
    );

    // `banking_stage::new_num_threads()` is parametrized over the `transaction_struct`, allowing it
    // to accept either `SanitizedTransaction` or `ResolvedTransactionView`.  But the adversarial
    // scheduler only produces the `SanitizedTransaction`, so making this code generic over both is
    // more work than necessary.

    let (non_vote_exit_signal, non_vote_thread_hdls) = spawn_adversarial_scheduler_and_workers(
        decision_maker,
        committer,
        poh_recorder,
        transaction_recorder,
        num_workers,
        log_messages_bytes_limit,
        replay_attack_receiver,
        drop_packets,
    );

    AdversarialBankingStage {
        non_vote_exit_signal,
        non_vote_thread_hdls,
    }
}

// NOTE This is mostly copied from `BankingStage::spawn_scheduler_and_workers()`.
#[allow(clippy::too_many_arguments)]
fn spawn_adversarial_scheduler_and_workers(
    decision_maker: DecisionMaker,
    committer: Committer,
    poh_recorder: &Arc<RwLock<PohRecorder>>,
    transaction_recorder: TransactionRecorder,
    num_workers: NonZeroUsize,
    log_messages_bytes_limit: Option<usize>,
    replay_attack_receiver: ReplayAttackReceiver,
    drop_packets: Arc<AtomicBool>,
) -> (Arc<AtomicBool>, Vec<JoinHandle<()>>) {
    // Create channels for communication between scheduler and workers
    let num_workers = num_workers.get();
    let (work_senders, work_receivers): (Vec<Sender<_>>, Vec<Receiver<_>>) =
        (0..num_workers).map(|_| bounded(10)).unzip();
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

    // Scheduler thread.
    //
    // `AttackScheduler` will stop when any of the connected worker channels are disconnected, so it
    // does not need a separate exit flag.
    let scheduler = AttackScheduler::new(
        decision_maker,
        work_senders,
        finished_work_receiver,
        num_workers,
        replay_attack_receiver,
        drop_packets,
    );

    non_vote_thread_hdls.push(
        Builder::new()
            .name("solAdvBnkTxSched".to_string())
            .spawn(move || scheduler.run())
            .unwrap(),
    );

    (exit, non_vote_thread_hdls)
}
