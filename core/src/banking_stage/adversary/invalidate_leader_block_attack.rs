//! Directly records invalid transaction(s), skipping execution & committing.
//!

use {
    solana_adversary::{
        adversary_feature_set, adversary_feature_set::invalidate_leader_block::InvalidationKind,
    },
    solana_clock::{DEFAULT_MS_PER_SLOT, DEFAULT_TICKS_PER_SLOT},
    solana_keypair::Keypair,
    solana_poh::{
        poh_recorder::{PohRecorder, SharedLeaderState},
        transaction_recorder::TransactionRecorder,
    },
    solana_pubkey::Pubkey,
    solana_runtime::bank::Bank,
    solana_signature::Signature,
    solana_system_transaction as system_transaction,
    std::{
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc, RwLock,
        },
        time::Duration,
    },
};

// This value was empirically derived by running this attack case and observing
// invalidation success rate. 100% success rate was observed with this value.
const END_OF_BLOCK_TICK_MARGIN: u64 = 2;

pub struct InvalidateLeaderBlockAttack {
    transaction_recorder: TransactionRecorder,
    shared_leader_state: SharedLeaderState,
    exit: Arc<AtomicBool>,
}

impl InvalidateLeaderBlockAttack {
    pub fn spawn(
        poh_recorder: &RwLock<PohRecorder>,
        transaction_recorder: TransactionRecorder,
        exit: Arc<AtomicBool>,
    ) -> std::thread::JoinHandle<()> {
        let poh_recorder = poh_recorder.read().unwrap();
        let shared_leader_state = poh_recorder.shared_leader_state();
        let invalid_transaction_recorder = InvalidateLeaderBlockAttack {
            transaction_recorder,
            shared_leader_state,
            exit,
        };

        std::thread::Builder::new()
            .name("solInvTxRec".to_string())
            .spawn(move || invalid_transaction_recorder.run())
            .unwrap()
    }

    fn run(self) {
        while !self.exit.load(Ordering::Relaxed) {
            let leader_bank = {
                let leader_state = self.shared_leader_state.load();
                match leader_state.working_bank().map(Arc::clone) {
                    Some(bank) => bank,
                    None => continue,
                }
            };

            let config = adversary_feature_set::invalidate_leader_block::get_config();
            let Some(invalidation_kind) = config.invalidation_kind else {
                continue;
            };

            // Wait until near the end of the block.
            let target_tick_height = leader_bank.max_tick_height();
            loop {
                match target_tick_height.checked_sub(leader_bank.tick_height()) {
                    None | Some(0) => break, // Bank is complete, or already at target height.
                    Some(tick_distance_from_target) => {
                        if tick_distance_from_target <= END_OF_BLOCK_TICK_MARGIN {
                            break;
                        }
                        // Target sleep duration is half the time to the target
                        // tick height so that we don't miss our invalidation
                        // window. Ticks are only a rough approximation of time,
                        // so using precise translation results in missing a lot
                        // of invalidations.
                        let wait_duration = Duration::from_millis(
                            tick_distance_from_target * DEFAULT_MS_PER_SLOT
                                / DEFAULT_TICKS_PER_SLOT
                                / 2,
                        );
                        std::thread::sleep(wait_duration);
                    }
                }
            }

            // Re-check bank is not complete, and skip if it is.
            // The slot may be missed for several reasons:
            // 1) Initial wake-up from `leader_bank_notifier` was significantly delayed.
            // 2) Taking the lock in `invalidate_leader_block::get_config()` was significantly
            //    delayed.
            // 3) The bank was completed while waiting for the target tick height - this is expected
            //    if not enough threads are available, causing inaccuracy in `sleep`.
            if leader_bank.is_complete() {
                warn!(
                    "Invalidation skipped for slot {} - bank is complete.\nThis is expected if \
                     not enough threads are available.",
                    leader_bank.slot()
                );
                continue;
            }

            match invalidation_kind {
                InvalidationKind::InvalidFeePayer => {
                    self.record_invalid_fee_payer(&leader_bank);
                }
                InvalidationKind::InvalidSignature => {
                    self.record_invalid_signature(&leader_bank);
                }
            }

            // wait for bank to complete
            while !leader_bank.is_complete() {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }

    /// Record a simple transfer transaction with an invalid fee-payer.
    fn record_invalid_fee_payer(&self, bank: &Bank) {
        let payer = Keypair::new();
        let receiver = Pubkey::new_unique();
        let transaction = system_transaction::transfer(&payer, &receiver, 1, bank.last_blockhash());

        let transactions = vec![transaction.into()];
        let summary = self
            .transaction_recorder
            .record_transactions(bank.slot(), transactions);
        if let Err(err) = summary.result {
            warn!(
                "Failed slot {} invalidation - invalid fee-payer transaction: {}",
                bank.slot(),
                err
            );
        } else {
            info!(
                "Invalidated slot {}: recorded invalid fee-payer transaction",
                bank.slot()
            );
        }
    }

    /// Record a simple transfer transaction with an invalid signature.
    fn record_invalid_signature(&self, bank: &Bank) {
        let payer = Keypair::new();
        let receiver = Pubkey::new_unique();
        let mut transaction =
            system_transaction::transfer(&payer, &receiver, 1, bank.last_blockhash());
        transaction.signatures[0] = Signature::new_unique();

        let transactions = vec![transaction.into()];
        let summary = self
            .transaction_recorder
            .record_transactions(bank.slot(), transactions);
        if let Err(err) = summary.result {
            warn!(
                "Failed slot {} invalidation - invalid signature transaction: {}",
                bank.slot(),
                err
            );
        } else {
            info!(
                "Invalidated slot {}: recorded invalid signature transaction",
                bank.slot()
            );
        }
    }
}
