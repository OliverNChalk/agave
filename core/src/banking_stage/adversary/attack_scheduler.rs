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
    crossbeam_channel::{Receiver, Sender, TryRecvError},
    solana_adversary::ReplayAttackReceiver,
    solana_runtime_transaction::runtime_transaction::RuntimeTransaction,
    solana_transaction::sanitized::SanitizedTransaction,
    std::{
        sync::{
            atomic::{self, AtomicBool},
            Arc,
        },
        thread::sleep,
        time::Duration,
    },
};

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
        } = self;

        'scheduler_loop: loop {
            match replay_attack_receiver.try_recv() {
                Ok(selected_attack) => {
                    info!("Reset selected generator to: {selected_attack:?}");
                    active_generator =
                        ActiveGenerator::with_selected_attack(selected_attack, num_workers);
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

            let num_generator_exec = active_generator.get_num_generator_exec_batch_size();
            let use_failed_transaction_hotpath = active_generator.use_failed_transaction_hotpath();

            // Batch transactions to amortize decision cost.
            for _ in 0..num_generator_exec {
                let (transactions, worker_index) = active_generator.generate_transactions(bank);

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
                    max_ages: (0..tx_count)
                        .map(|_| scheduler_messages::MaxAge::MAX)
                        .collect(),
                    use_failed_transaction_hotpath,
                };

                consume_work_senders[worker_index]
                    .send(scheduled_work)
                    .unwrap();
            }
        }
    }
}
