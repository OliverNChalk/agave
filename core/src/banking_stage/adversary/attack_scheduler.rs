//! Simple scheduler that drops network packets and generates transactions.
//!     - this is useful for testing the banking stage without the network
//!       or in creating stress-tests on a local network.

use {
    crate::banking_stage::{
        adversary::transaction_generators::ActiveGenerator,
        decision_maker::DecisionMaker,
        scheduler_messages::{self, ConsumeWork, FinishedConsumeWork},
        transaction_scheduler::batch_id_generator::BatchIdGenerator,
    },
    crossbeam_channel::{Receiver, SendError, Sender, TryRecvError},
    rayon::{ThreadPool, ThreadPoolBuilder},
    solana_adversary::ReplayAttackReceiver,
    solana_runtime::bank::Bank,
    solana_runtime_transaction::runtime_transaction::RuntimeTransaction,
    solana_transaction::sanitized::SanitizedTransaction,
    std::{
        ops::ControlFlow,
        sync::{
            atomic::{self, AtomicBool},
            Arc,
        },
        thread::sleep,
        time::{Duration, Instant},
    },
};

const RAYON_THREADS_POOL_SIZE_FOR_TX_GENERATION: usize = 2;

/// This is an adversarial scheduler.  It does not process any incoming transactions, but instead
/// will generate transactions using our test generators.
///
// It is similar to `banking_stage::transaction_scheduler::greedy_scheduler::GreedyScheduler`.
// Mostly the `SchedulingCommon` part.
pub struct AttackScheduler {
    /// Decision maker - only generate when leader
    decision_maker: DecisionMaker,

    /// To BankingStageWorker(s)
    consume_work_senders: Vec<Sender<ConsumeWork<RuntimeTransaction<SanitizedTransaction>>>>,

    /// From BankingStageWorker
    finished_consume_work_receiver:
        Receiver<FinishedConsumeWork<RuntimeTransaction<SanitizedTransaction>>>,

    /// Number of workers replaying transactions
    num_workers: usize,

    /// Informs the `AttackScheduler` of the attack it should start running.
    replay_attack_receiver: ReplayAttackReceiver,

    /// Controls the `BankingStage` scheduler, indicating that it should be dropping packets rather
    /// then processing them.
    drop_packets: Arc<AtomicBool>,

    /// Transaction generators' thread pool.
    tx_generator_thread_pool: Arc<ThreadPool>,
}

impl AttackScheduler {
    pub fn new(
        decision_maker: DecisionMaker,
        consume_work_senders: Vec<Sender<ConsumeWork<RuntimeTransaction<SanitizedTransaction>>>>,
        finished_consume_work_receiver: Receiver<
            FinishedConsumeWork<RuntimeTransaction<SanitizedTransaction>>,
        >,
        num_workers: usize,
        replay_attack_receiver: ReplayAttackReceiver,
        drop_packets: Arc<AtomicBool>,
    ) -> Self {
        Self {
            decision_maker,
            consume_work_senders,
            finished_consume_work_receiver,
            num_workers,
            replay_attack_receiver,
            drop_packets,
            tx_generator_thread_pool: Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(RAYON_THREADS_POOL_SIZE_FOR_TX_GENERATION)
                    .thread_name(|i| format!("solAdvTxGen{i}"))
                    .build()
                    .unwrap(),
            ),
        }
    }

    pub fn run(self) {
        let mut tx_batch_id_gen = BatchIdGenerator::default();
        let mut tx_id: usize = 0;
        let mut active_generator = None;

        let Self {
            decision_maker,
            consume_work_senders,
            finished_consume_work_receiver,
            num_workers,
            replay_attack_receiver,
            drop_packets,
            tx_generator_thread_pool,
        } = self;

        'scheduler_loop: loop {
            match replay_attack_receiver.try_recv() {
                Ok(selected_attack) => {
                    info!("Reset selected generator to: {selected_attack:?}");
                    active_generator = ActiveGenerator::given(
                        selected_attack,
                        num_workers,
                        Arc::clone(&tx_generator_thread_pool),
                    );
                }
                Err(TryRecvError::Empty) => {
                    // continue executing active_generator
                }
                Err(TryRecvError::Disconnected) => {
                    debug!("AttackScheduler exiting");
                    break 'scheduler_loop;
                }
            }

            // Drop finished work from banking stage so we don't OOM
            finished_consume_work_receiver.try_iter().for_each(drop);

            let Some(ref mut active_generator) = active_generator else {
                // We do not want to drop traffic if no attack is being run.
                // We do not want to drop traffic if no attack is being run.
                if drop_packets.load(atomic::Ordering::Relaxed) {
                    drop_packets.store(false, atomic::Ordering::Relaxed);
                }

                sleep(Duration::from_millis(100));
                continue;
            };

            if !drop_packets.load(atomic::Ordering::Relaxed) {
                drop_packets.store(true, atomic::Ordering::Relaxed);
            }

            let decision = decision_maker.make_consume_or_forward_decision();

            let Some(bank) = decision.bank() else {
                // If we do not have a leader slot, then we are not the leader and we are not going
                // to process any traffic.  Wait a bit to avoid blocking the CPU.
                sleep(Duration::from_millis(1));
                continue;
            };

            let keep_running = generate_attack_transactions(
                &mut tx_batch_id_gen,
                &mut tx_id,
                active_generator,
                bank.as_ref(),
                &consume_work_senders,
            );

            if keep_running.is_break() {
                break 'scheduler_loop;
            }
        }
    }
}

fn generate_attack_transactions(
    tx_batch_id_gen: &mut BatchIdGenerator,
    tx_id: &mut usize,
    active_generator: &mut ActiveGenerator,
    bank: &Bank,
    consume_work_senders: &[Sender<ConsumeWork<RuntimeTransaction<SanitizedTransaction>>>],
) -> ControlFlow<()> {
    // There is an assumption that as some transaction generators are very fast, we may want to run
    // them in a loop more than once, before we go back to checking all the other conditions the
    // main loop is supposed to react to.
    //
    // We did see a difference in the number of transactions we managed to pack into a single block
    // with the simple transfer attack, with and without this optimization.  But at that point the
    // loop was doing more operations than now.  So it is possible that right now this amortization
    // is not providing any additional value.  It is also the case, that the other operations in
    // this loop are either very cheap, or are meaningful for the active generator.
    //
    // In any case, we could afford to block here for a bit, before we go back to checking all the
    // other conditions in the loop above.  10ms is a rather random number that is not too long
    // considering the slot time of 400ms.  We mostly expect attacks to operate at a scale of at
    // least 4 blocks - a single leader schedule item.
    const NON_STOP_GENERATION_INTERVAL: Duration = Duration::from_millis(10);
    let start = Instant::now();

    let use_failed_transaction_hotpath = active_generator.use_failed_transaction_hotpath();
    let use_invalid_fee_payer = active_generator.use_invalid_fee_payer();

    loop {
        let (transactions, worker_index) = active_generator.generate_transactions(bank);

        let scheduled_work = transactions_to_scheduled_work(
            tx_batch_id_gen,
            tx_id,
            use_failed_transaction_hotpath,
            use_invalid_fee_payer,
            transactions,
        );
        match consume_work_senders[worker_index].send(scheduled_work) {
            Ok(()) => (),
            Err(SendError(_)) => {
                // The channel was disconnected.
                return ControlFlow::Break(());
            }
        }

        // We want to run the generator at least once, so this check must be last in the loop.
        if start.elapsed() >= NON_STOP_GENERATION_INTERVAL {
            break;
        }
    }

    ControlFlow::Continue(())
}

fn transactions_to_scheduled_work(
    tx_batch_id_gen: &mut BatchIdGenerator,
    tx_id: &mut usize,
    use_failed_transaction_hotpath: bool,
    use_invalid_fee_payer: bool,
    transactions: Vec<SanitizedTransaction>,
) -> ConsumeWork<RuntimeTransaction<SanitizedTransaction>> {
    let transactions = transactions
        .into_iter()
        .map(|tx| {
            RuntimeTransaction::new_from_sanitized(tx).unwrap_or_else(|err| {
                panic!(
                    "Failed to convert a `SanitizedTransaction` produced by the generator into a \
                     `RuntimeTransaction<SanitizedTransaction>`.\nError: {err}",
                )
            })
        })
        .collect::<Vec<_>>();

    let tx_count = transactions.len();

    ConsumeWork::<RuntimeTransaction<SanitizedTransaction>> {
        batch_id: tx_batch_id_gen.next(),
        ids: (0..tx_count)
            .map(|_| {
                *tx_id += 1;
                *tx_id
            })
            .collect(),
        transactions,
        max_ages: (0..tx_count)
            .map(|_| scheduler_messages::MaxAge::MAX)
            .collect(),
        use_failed_transaction_hotpath,
        use_invalid_fee_payer,
    }
}
