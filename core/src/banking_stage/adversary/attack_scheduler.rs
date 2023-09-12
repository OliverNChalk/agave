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
    crossbeam_channel::{Receiver, Sender},
    solana_adversary::adversary_feature_set::replay_stage_attack,
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

    /// Selected generator
    active_generator: ActiveGenerator,

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
        active_generator: ActiveGenerator,
        drop_packets: Arc<AtomicBool>,
    ) -> Self {
        Self {
            decision_maker,
            consume_work_senders,
            finished_consume_work_receiver,
            active_generator,
            drop_packets,
        }
    }

    pub fn run(self, exit: Arc<AtomicBool>) {
        let mut tx_batch_id_gen = BatchIdGenerator::default();
        let mut tx_id: usize = 0;

        let Self {
            decision_maker,
            consume_work_senders,
            finished_consume_work_receiver,
            mut active_generator,
            drop_packets,
        } = self;

        'scheduler_loop: loop {
            if exit.load(atomic::Ordering::Relaxed) {
                debug!("AttackScheduler exiting");
                break 'scheduler_loop;
            }

            // Drop finished work from banking stage so we don't OOM
            finished_consume_work_receiver.try_iter().for_each(drop);

            update_active_attack(&mut active_generator);
            if !active_generator.is_active() {
                // We do not want to drop traffic if no attack is being run.
                if drop_packets.load(atomic::Ordering::Relaxed) {
                    drop_packets.store(false, atomic::Ordering::Relaxed);
                }

                sleep(Duration::from_millis(100));
                continue;
            }

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

            let num_tx_batches = active_generator.get_num_tx_batches();
            // Batch transactions to amortize decision cost.
            for _ in 0..num_tx_batches {
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
                    use_failed_transaction_hotpath: false,
                };

                consume_work_senders[worker_index]
                    .send(scheduled_work)
                    .unwrap();
            }
        }
    }
}

fn update_active_attack(active_generator: &mut ActiveGenerator) {
    let replay_stage_attack::AdversarialConfig {
        selected_attack, ..
    } = replay_stage_attack::get_config();
    active_generator.ensure_active(selected_attack);
}
