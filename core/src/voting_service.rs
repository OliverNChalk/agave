use {
    crate::{
        consensus::tower_storage::{SavedTowerVersions, TowerStorage},
        mock_alpenglow_consensus::MockAlpenglowConsensus,
        next_leader::upcoming_leader_tpu_vote_sockets,
    },
    bincode::serialize,
    crossbeam_channel::Receiver,
    solana_adversary::adversary_feature_set::delay_votes,
    solana_client::connection_cache::ConnectionCache,
    solana_clock::{Slot, FORWARD_TRANSACTIONS_TO_LEADER_AT_SLOT_OFFSET},
    solana_connection_cache::client_connection::ClientConnection,
    solana_gossip::{cluster_info::ClusterInfo, epoch_specs::EpochSpecs},
    solana_measure::measure::Measure,
    solana_poh::poh_recorder::PohRecorder,
    solana_runtime::bank_forks::BankForks,
    solana_transaction::Transaction,
    solana_transaction_error::TransportError,
    std::{
        net::{SocketAddr, UdpSocket},
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, RwLock,
        },
        thread::{self, Builder, JoinHandle},
    },
    thiserror::Error,
};

pub enum VoteOp {
    PushVote {
        tx: Transaction,
        tower_slots: Vec<Slot>,
        saved_tower: SavedTowerVersions,
    },
    RefreshVote {
        tx: Transaction,
        last_voted_slot: Slot,
    },
}

impl VoteOp {
    fn tx(&self) -> &Transaction {
        match self {
            VoteOp::PushVote { tx, .. } => tx,
            VoteOp::RefreshVote { tx, .. } => tx,
        }
    }
}

#[derive(Debug, Error)]
enum SendVoteError {
    #[error(transparent)]
    BincodeError(#[from] bincode::Error),
    #[error("Invalid TPU address")]
    InvalidTpuAddress,
    #[error(transparent)]
    TransportError(#[from] TransportError),
}

fn send_vote_transaction(
    cluster_info: &ClusterInfo,
    transaction: &Transaction,
    tpu: Option<SocketAddr>,
    connection_cache: &Arc<ConnectionCache>,
) -> Result<(), SendVoteError> {
    let tpu = tpu
        .or_else(|| {
            cluster_info
                .my_contact_info()
                .tpu(connection_cache.protocol())
        })
        .ok_or(SendVoteError::InvalidTpuAddress)?;
    let buf = Arc::new(serialize(transaction)?);
    let client = connection_cache.get_connection(&tpu);

    client.send_data_async(buf).map_err(|err| {
        error!("Ran into an error when sending vote: {err:?} to {tpu:?}");
        SendVoteError::from(err)
    })
}

pub struct VotingService {
    thread_hdl: JoinHandle<()>,
    vote_storage_size: Arc<AtomicUsize>,
}

impl VotingService {
    fn run(
        vote_storage_size: Arc<AtomicUsize>,
        vote_receiver: Receiver<VoteOp>,
        cluster_info: Arc<ClusterInfo>,
        poh_recorder: Arc<RwLock<PohRecorder>>,
        tower_storage: Arc<dyn TowerStorage>,
        connection_cache: Arc<ConnectionCache>,
        bank_forks: Arc<RwLock<BankForks>>,
        mut mock_alpenglow: Option<MockAlpenglowConsensus>,
    ) {
        let mut vote_storage = Vec::new();
        loop {
            if vote_storage.is_empty() {
                match vote_receiver.recv() {
                    Ok(vote) => vote_storage.push(vote),
                    Err(_) => break,
                }
            } else {
                // Only wait for a short time so we can handle stored
                // votes if we're allowed.
                match vote_receiver.recv_timeout(std::time::Duration::from_millis(10)) {
                    Ok(vote) => vote_storage.push(vote),
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => (),
                    Err(_) => break,
                }
            }

            // Floor to 0 to allow all votes to go through
            let current_slot = poh_recorder
                .read()
                .unwrap()
                .leader_and_slot_after_n_slots(0)
                .map_or(0, |(_, current_slot)| current_slot);

            let delay_votes_by_slot_count = delay_votes::get_config().delay_votes_by_slot_count;

            vote_storage.retain(|vote_op| {
                // Figure out if we are casting a vote for a new slot, and what slot it is for
                let vote_slot = match vote_op {
                    VoteOp::PushVote { tower_slots, .. } => {
                        if let Some(vote_slot) = tower_slots.last() {
                            if *vote_slot + delay_votes_by_slot_count > current_slot {
                                debug!(
                                    "Not handling vote for slot {vote_slot}, currently at slot \
                                     {current_slot} and delaying votes by \
                                     {delay_votes_by_slot_count}"
                                );
                                return true;
                            }
                            Some(*vote_slot)
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                Self::handle_vote(
                    &cluster_info,
                    &poh_recorder,
                    tower_storage.as_ref(),
                    vote_op,
                    connection_cache.clone(),
                );

                // trigger mock alpenglow vote if we have just cast an actual vote
                if let Some(slot) = vote_slot {
                    if let Some(ag) = mock_alpenglow.as_mut() {
                        let root_bank = { bank_forks.read().unwrap().root_bank() };
                        ag.signal_new_slot(slot, &root_bank);
                    }
                }

                false
            });
            vote_storage_size.store(vote_storage.len(), Ordering::Relaxed);
        }
        if let Some(ag) = mock_alpenglow {
            let _ = ag.join();
        }
    }
    pub fn new(
        vote_receiver: Receiver<VoteOp>,
        cluster_info: Arc<ClusterInfo>,
        poh_recorder: Arc<RwLock<PohRecorder>>,
        tower_storage: Arc<dyn TowerStorage>,
        connection_cache: Arc<ConnectionCache>,
        alpenglow_socket: Option<UdpSocket>,
        bank_forks: Arc<RwLock<BankForks>>,
    ) -> Self {
        let vote_storage_size = Arc::new(AtomicUsize::new(0));
        let thread_hdl = Builder::new()
            .name("solVoteService".to_string())
            .spawn({
                let mock_alpenglow = alpenglow_socket.map(|s| {
                    MockAlpenglowConsensus::new(
                        s,
                        cluster_info.clone(),
                        EpochSpecs::from(bank_forks.clone()),
                    )
                });
                let vote_storage_size = vote_storage_size.clone();
                move || {
                    Self::run(
                        vote_storage_size,
                        vote_receiver,
                        cluster_info,
                        poh_recorder,
                        tower_storage,
                        connection_cache.clone(),
                        bank_forks,
                        mock_alpenglow,
                    )
                }
            })
            .unwrap();
        Self {
            thread_hdl,
            vote_storage_size,
        }
    }

    pub fn vote_storage_count(&self) -> usize {
        self.vote_storage_size.load(Ordering::Relaxed)
    }

    pub fn handle_vote(
        cluster_info: &ClusterInfo,
        poh_recorder: &RwLock<PohRecorder>,
        tower_storage: &dyn TowerStorage,
        vote_op: &VoteOp,
        connection_cache: Arc<ConnectionCache>,
    ) {
        if let VoteOp::PushVote { saved_tower, .. } = &vote_op {
            let mut measure = Measure::start("tower storage save");
            if let Err(err) = tower_storage.store(saved_tower) {
                error!("Unable to save tower to storage: {err:?}");
                std::process::exit(1);
            }
            measure.stop();
            trace!("{measure}");
        }

        // Attempt to send our vote transaction to the leaders for the next few
        // slots. From the current slot to the forwarding slot offset
        // (inclusive).
        const UPCOMING_LEADER_FANOUT_SLOTS: u64 =
            FORWARD_TRANSACTIONS_TO_LEADER_AT_SLOT_OFFSET.saturating_add(1);
        #[cfg(test)]
        static_assertions::const_assert_eq!(UPCOMING_LEADER_FANOUT_SLOTS, 3);
        let upcoming_leader_sockets = upcoming_leader_tpu_vote_sockets(
            cluster_info,
            poh_recorder,
            UPCOMING_LEADER_FANOUT_SLOTS,
            connection_cache.protocol(),
        );

        if !upcoming_leader_sockets.is_empty() {
            for tpu_vote_socket in upcoming_leader_sockets {
                let _ = send_vote_transaction(
                    cluster_info,
                    vote_op.tx(),
                    Some(tpu_vote_socket),
                    &connection_cache,
                );
            }
        } else {
            // Send to our own tpu vote socket if we cannot find a leader to send to
            let _ = send_vote_transaction(cluster_info, vote_op.tx(), None, &connection_cache);
        }

        match vote_op {
            VoteOp::PushVote {
                tx, tower_slots, ..
            } => {
                cluster_info.push_vote(tower_slots, tx.clone());
            }
            VoteOp::RefreshVote {
                tx,
                last_voted_slot,
            } => {
                cluster_info.refresh_vote(tx.clone(), *last_voted_slot);
            }
        }
    }

    pub fn join(self) -> thread::Result<()> {
        self.thread_hdl.join()
    }
}
