#![allow(clippy::rc_buffer)]

use {
    super::{
        broadcast_utils::{self, ReceiveResults},
        *,
    },
    crate::cluster_nodes::ClusterNodesCache,
    solana_adversary::adversary_feature_set::send_duplicate_blocks::AdversarialConfig,
    solana_entry::entry::Entry,
    solana_hash::Hash,
    solana_keypair::Keypair,
    solana_ledger::shred::{
        ProcessShredsStats, ReedSolomonCache, Shred, ShredType, Shredder, MAX_CODE_SHREDS_PER_SLOT,
        MAX_DATA_SHREDS_PER_SLOT,
    },
    solana_runtime::bank::Bank,
    solana_system_transaction as system_transaction,
    solana_time_utils::AtomicInterval,
    std::{borrow::Cow, sync::RwLock},
    tokio::sync::mpsc::Sender as AsyncSender,
};

#[derive(Clone)]
pub struct StandardBroadcastRun {
    slot: Slot,
    parent: Slot,
    chained_merkle_root: Hash,
    carryover_entry: Option<WorkingBankEntry>,
    next_shred_index: u32,
    next_code_index: u32,
    // If last_tick_height has reached bank.max_tick_height() for this slot
    // and so the slot is completed and all shreds are already broadcast.
    completed: bool,
    process_shreds_stats: ProcessShredsStats,
    transmit_shreds_stats: Arc<Mutex<SlotBroadcastStats<TransmitShredsStats>>>,
    insert_shreds_stats: Arc<Mutex<SlotBroadcastStats<InsertShredsStats>>>,
    slot_broadcast_start: Instant,
    shred_version: u16,
    last_datapoint_submit: Arc<AtomicInterval>,
    num_batches: usize,
    cluster_nodes_cache: Arc<ClusterNodesCache<BroadcastStage>>,
    reed_solomon_cache: Arc<ReedSolomonCache>,
    recent_blockhash: Option<Hash>,
    // To separate shred generation from shred transmission so we can simulate
    // different timing involved, we save all generated shreds in shreds_to_send,
    // so we can actually send out them out in any order and with any delay.
    shreds_to_send: Vec<BroadcastShredsData>,
}

#[derive(Clone, Debug)]
struct BroadcastShredsData {
    data_shreds: Arc<Vec<Shred>>,
    coding_shreds: Arc<Vec<Shred>>,
    batch_info: Option<BroadcastShredBatchInfo>,
    send_after: Instant,
}

fn send_shreds(
    data: &BroadcastShredsData,
    socket_sender: &Sender<(Arc<Vec<Shred>>, Option<BroadcastShredBatchInfo>)>,
    blockstore_sender: &Sender<(Arc<Vec<Shred>>, Option<BroadcastShredBatchInfo>)>,
) -> Result<()> {
    // Send data shreds
    socket_sender.send((data.data_shreds.clone(), data.batch_info.clone()))?;
    blockstore_sender.send((data.data_shreds.clone(), data.batch_info.clone()))?;

    // Send coding shreds
    socket_sender.send((data.coding_shreds.clone(), data.batch_info.clone()))?;
    blockstore_sender.send((data.coding_shreds.clone(), data.batch_info.clone()))?;

    Ok(())
}

#[derive(Debug)]
enum BroadcastError {
    TooManyShreds,
}

/// Holds shredding progress for a single slot.
///
/// Used by duplicate block attacks in order to be able to rollback the shredding process to the
/// beginning of the current batch of entries and produce a new set of shreds with some differences.
#[derive(Clone, Copy)]
struct SlotShreddingProgress {
    chained_merkle_root: Hash,
    next_data_shred_index: u32,
    next_code_shred_index: u32,
}

impl SlotShreddingProgress {
    fn record(from: &StandardBroadcastRun) -> Self {
        Self {
            chained_merkle_root: from.chained_merkle_root,
            next_data_shred_index: from.next_shred_index,
            next_code_shred_index: from.next_code_index,
        }
    }
}

impl From<&SlotShreddingProgress> for (Hash, u32, u32) {
    fn from(
        SlotShreddingProgress {
            chained_merkle_root,
            next_data_shred_index,
            next_code_shred_index,
        }: &SlotShreddingProgress,
    ) -> Self {
        (
            *chained_merkle_root,
            *next_data_shred_index,
            *next_code_shred_index,
        )
    }
}

impl StandardBroadcastRun {
    pub(super) fn new(shred_version: u16) -> Self {
        let cluster_nodes_cache = Arc::new(ClusterNodesCache::<BroadcastStage>::new(
            CLUSTER_NODES_CACHE_NUM_EPOCH_CAP,
            CLUSTER_NODES_CACHE_TTL,
        ));
        Self {
            slot: Slot::MAX,
            parent: Slot::MAX,
            chained_merkle_root: Hash::default(),
            carryover_entry: None,
            next_shred_index: 0,
            next_code_index: 0,
            completed: true,
            process_shreds_stats: ProcessShredsStats::default(),
            transmit_shreds_stats: Arc::default(),
            insert_shreds_stats: Arc::default(),
            slot_broadcast_start: Instant::now(),
            shred_version,
            last_datapoint_submit: Arc::default(),
            num_batches: 0,
            cluster_nodes_cache,
            reed_solomon_cache: Arc::<ReedSolomonCache>::default(),
            recent_blockhash: None,
            shreds_to_send: Vec::default(),
        }
    }

    // If the current slot has changed, generates an empty shred indicating
    // last shred in the previous slot, along with coding shreds for the data
    // shreds buffered.
    fn finish_prev_slot(
        &mut self,
        keypair: &Keypair,
        max_ticks_in_slot: u8,
        stats: &mut ProcessShredsStats,
    ) -> Vec<Shred> {
        if self.completed {
            return vec![];
        }
        // Set the reference_tick as if the PoH completed for this slot
        let reference_tick = max_ticks_in_slot;
        let shreds: Vec<_> =
            Shredder::new(self.slot, self.parent, reference_tick, self.shred_version)
                .unwrap()
                .make_merkle_shreds_from_entries(
                    keypair,
                    &[],  // entries
                    true, // is_last_in_slot,
                    self.chained_merkle_root,
                    self.next_shred_index,
                    self.next_code_index,
                    &self.reed_solomon_cache,
                    stats,
                )
                .inspect(|shred| stats.record_shred(shred))
                .collect();
        if let Some(shred) = shreds.iter().max_by_key(|shred| shred.fec_set_index()) {
            self.chained_merkle_root = shred.merkle_root().unwrap();
        }
        self.report_and_reset_stats(/*was_interrupted:*/ true);
        self.completed = true;
        shreds
    }

    #[allow(clippy::too_many_arguments)]
    fn entries_to_shreds(
        &mut self,
        config: &AdversarialConfig,
        keypair: &Keypair,
        entries: &[Entry],
        reference_tick: u8,
        is_slot_end: bool,
        process_stats: &mut ProcessShredsStats,
        max_data_shreds_per_slot: u32,
        max_code_shreds_per_slot: u32,
        validator_index: usize,
        bank: Arc<Bank>,
        send_after: Instant,
        slot_progress_at_batch_start: &SlotShreddingProgress,
    ) -> std::result::Result<Option<BroadcastShredsData>, BroadcastError> {
        let (chained_merkle_root, next_shred_index, next_code_index) = if validator_index > 0 {
            slot_progress_at_batch_start.into()
        } else {
            (
                self.chained_merkle_root,
                self.next_shred_index,
                self.next_code_index,
            )
        };

        let shreds = Shredder::new(self.slot, self.parent, reference_tick, self.shred_version)
            .unwrap()
            .make_merkle_shreds_from_entries(
                keypair,
                entries,
                is_slot_end,
                chained_merkle_root,
                next_shred_index,
                next_code_index,
                &self.reed_solomon_cache,
                &mut self.process_shreds_stats,
            );

        let mut data_shreds = Vec::new();
        let mut coding_shreds = Vec::new();

        for shred in shreds {
            process_stats.record_shred(&shred);
            match shred.shred_type() {
                ShredType::Data => {
                    self.next_shred_index = self.next_shred_index.max(shred.index() + 1);
                    data_shreds.push(shred);
                }
                ShredType::Code => {
                    self.next_code_index = self.next_code_index.max(shred.index() + 1);
                    coding_shreds.push(shred);
                }
            }
        }

        if validator_index == 0 {
            // Update state only for the primary validator
            if let Some(shred) = data_shreds.iter().max_by_key(|shred| shred.index()) {
                self.chained_merkle_root = shred.merkle_root().unwrap();
                self.next_shred_index = shred.index() + 1;
            }
            if let Some(index) = coding_shreds.iter().map(Shred::index).max() {
                self.next_code_index = index + 1;
            }
        }

        if self.next_shred_index > max_data_shreds_per_slot {
            return Err(BroadcastError::TooManyShreds);
        }
        if self.next_code_index > max_code_shreds_per_slot {
            return Err(BroadcastError::TooManyShreds);
        }

        self.num_batches += 2; // Track number of batches processed
        let num_expected_batches = if is_slot_end {
            Some(self.num_batches)
        } else {
            None
        };

        let destinations = config.send_destinations.get(validator_index).cloned();
        Ok(Some(BroadcastShredsData {
            data_shreds: Arc::new(data_shreds),
            coding_shreds: Arc::new(coding_shreds),
            send_after,
            batch_info: Some(BroadcastShredBatchInfo {
                slot: bank.slot(),
                num_expected_batches,
                slot_start_ts: self.slot_broadcast_start,
                was_interrupted: false,
                destinations,
            }),
        }))
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn test_process_receive_results(
        &mut self,
        config: &AdversarialConfig,
        keypair: &Keypair,
        cluster_info: &ClusterInfo,
        sock: &UdpSocket,
        blockstore: &Blockstore,
        receive_results: ReceiveResults,
        bank_forks: &RwLock<BankForks>,
        quic_endpoint_sender: &AsyncSender<(SocketAddr, Bytes)>,
        expected_shred_batches_to_send: usize,
    ) -> Result<()> {
        let (bsend, brecv) = unbounded();
        let (ssend, srecv) = unbounded();

        let mut process_stats = ProcessShredsStats::default();
        self.process_receive_results(
            config,
            keypair,
            blockstore,
            &ssend,
            &bsend,
            receive_results,
            &mut process_stats,
        )?;

        assert_eq!(self.shreds_to_send.len(), expected_shred_batches_to_send);

        self.send_all_shreds(&ssend, &bsend, &mut process_stats)?;

        for _ in 0..expected_shred_batches_to_send {
            // Data
            let _ = self.transmit(
                &srecv,
                cluster_info,
                BroadcastSocket::Udp(sock),
                bank_forks,
                quic_endpoint_sender,
            );
            let _ = self.record(&brecv, blockstore);
            // Coding
            let _ = self.transmit(
                &srecv,
                cluster_info,
                BroadcastSocket::Udp(sock),
                bank_forks,
                quic_endpoint_sender,
            );
            let _ = self.record(&brecv, blockstore);
        }

        Ok(())
    }

    fn process_receive_results(
        &mut self,
        config: &AdversarialConfig,
        keypair: &Keypair,
        blockstore: &Blockstore,
        socket_sender: &Sender<(Arc<Vec<Shred>>, Option<BroadcastShredBatchInfo>)>,
        blockstore_sender: &Sender<(Arc<Vec<Shred>>, Option<BroadcastShredBatchInfo>)>,
        receive_results: ReceiveResults,
        process_stats: &mut ProcessShredsStats,
    ) -> Result<()> {
        let num_entries = receive_results.entries.len();
        let bank = receive_results.bank.clone();
        let last_tick_height = receive_results.last_tick_height;
        inc_new_counter_info!("broadcast_service-entries_received", num_entries);

        let mut to_shreds_time = Measure::start("broadcast_to_shreds");

        let now = Instant::now();

        // 1) We transitioned slots.
        if self.slot != bank.slot() {
            if !self.completed {
                let shreds =
                    self.finish_prev_slot(keypair, bank.ticks_per_slot() as u8, process_stats);
                debug_assert!(shreds.iter().all(|shred| shred.slot() == self.slot));
                let batch_info = Some(BroadcastShredBatchInfo {
                    slot: self.slot,
                    num_expected_batches: Some(self.num_batches + 1),
                    slot_start_ts: self.slot_broadcast_start,
                    was_interrupted: true,
                    destinations: None,
                });
                let shreds = Arc::new(shreds);
                socket_sender.send((shreds.clone(), batch_info.clone()))?;
                blockstore_sender.send((shreds, batch_info))?;
            }

            if blockstore
                .meta(bank.slot())
                .unwrap()
                .filter(|slot_meta| slot_meta.received > 0 || slot_meta.consumed > 0)
                .is_some()
            {
                process_stats.num_extant_slots += 1;
                return Err(Error::DuplicateSlotBroadcast(bank.slot()));
            }

            let chained_merkle_root = if self.slot == bank.parent_slot() {
                self.chained_merkle_root
            } else {
                broadcast_utils::get_chained_merkle_root_from_parent(
                    bank.slot(),
                    bank.parent_slot(),
                    blockstore,
                )
                .unwrap_or_else(|err: Error| {
                    error!("Unknown chained Merkle root: {err:?}");
                    process_stats.err_unknown_chained_merkle_root += 1;
                    Hash::default()
                })
            };

            self.slot = bank.slot();
            self.parent = bank.parent_slot();
            self.chained_merkle_root = chained_merkle_root;
            self.next_shred_index = 0u32;
            self.next_code_index = 0u32;
            self.completed = false;
            self.slot_broadcast_start = Instant::now();
            self.num_batches = 0;
            process_stats.receive_elapsed = 0;
            process_stats.coalesce_elapsed = 0;
        }

        let is_last_in_slot = last_tick_height == bank.max_tick_height();
        let reference_tick = last_tick_height
            .saturating_add(bank.ticks_per_slot())
            .saturating_sub(bank.max_tick_height());

        // Preserve invalidator logic: update recent blockhash from entries
        for entry in &receive_results.entries {
            if !entry.transactions.is_empty() {
                self.recent_blockhash = Some(*entry.transactions[0].message.recent_blockhash());
                break;
            }
        }

        // This delay is used in an attack where we try to partition the network by
        // delivering the first block of one leader simultaneously with the first block of
        // the next leader.
        let turbine_send_delay_time = now + Duration::from_millis(config.turbine_send_delay_ms);
        let send_after = if is_last_in_slot {
            // Delay last entry so that duplicate version can propagate first.
            turbine_send_delay_time + Duration::from_millis(config.send_original_after_ms)
        } else {
            turbine_send_delay_time
        };

        let slot_progress_at_batch_start = SlotShreddingProgress::record(self);

        if let Some(shred_data) = self
            .entries_to_shreds(
                config,
                keypair,
                &receive_results.entries,
                reference_tick as u8,
                is_last_in_slot,
                process_stats,
                MAX_DATA_SHREDS_PER_SLOT as u32,
                MAX_CODE_SHREDS_PER_SLOT as u32,
                0, // 0 means original blocks
                bank.clone(),
                send_after,
                &slot_progress_at_batch_start,
            )
            .expect("tried to generate a block that exceeds max shred count!")
        {
            if let Some(shred) = shred_data.data_shreds.first() {
                if shred.index() == 0 {
                    blockstore
                        .insert_cow_shreds(
                            [Cow::Borrowed(shred)],
                            None, // leader_schedule
                            true, // is_trusted
                        )
                        .expect("Failed to insert shreds in blockstore");
                }
            }
            self.shreds_to_send.push(shred_data);
        }

        if receive_results.entries.len() > 1 && is_last_in_slot && self.recent_blockhash.is_some() {
            let mut new_entries = receive_results.entries.clone();
            let original_entries_to_keep = std::cmp::max(
                1,
                new_entries
                    .len()
                    .saturating_sub(config.new_entry_index_from_end),
            );
            let send_after = turbine_send_delay_time;
            for validator_index in 0..config.num_duplicate_validators {
                // Cut original entries into two parts, we will insert transaction in between.
                let mut popped = new_entries.split_off(original_entries_to_keep);
                if validator_index > 0 {
                    popped.remove(0);
                }
                new_entries.push(Self::calculate_duplicate_blocks_suffix(
                    &mut popped,
                    validator_index == 0 && bank.hashes_per_tick().unwrap_or(0) > 0,
                    new_entries.last().unwrap().hash,
                    self.recent_blockhash.unwrap(),
                    keypair,
                ));
                new_entries.extend(popped);

                if let Some(shred_data) = self
                    .entries_to_shreds(
                        config,
                        keypair,
                        &new_entries,
                        reference_tick as u8,
                        is_last_in_slot,
                        process_stats,
                        MAX_DATA_SHREDS_PER_SLOT as u32,
                        MAX_CODE_SHREDS_PER_SLOT as u32,
                        validator_index + 1, // Duplicates start from index 1
                        bank.clone(),
                        send_after,
                        &slot_progress_at_batch_start,
                    )
                    .expect("tried to generate a block that exceeds max shred count!")
                {
                    self.shreds_to_send.push(shred_data);
                }
            }
        }

        to_shreds_time.stop();
        let mut get_leader_schedule_time = Measure::start("broadcast_get_leader_schedule");
        get_leader_schedule_time.stop();

        process_stats.shredding_elapsed = to_shreds_time.as_us();
        process_stats.get_leader_schedule_elapsed = get_leader_schedule_time.as_us();

        self.process_shreds_stats += *process_stats;

        if is_last_in_slot {
            self.report_and_reset_stats(false);
            self.completed = true;
        }
        Ok(())
    }

    fn calculate_duplicate_blocks_suffix(
        original_suffix: &mut Vec<Entry>,
        mut should_decrease_hashes: bool,
        prev_entry_hash: Hash,
        recent_block_hash: Hash,
        keypair: &Keypair,
    ) -> Entry {
        // Add a new random transaction to make each duplicate block different.
        let extra_tx =
            system_transaction::transfer(keypair, &Pubkey::new_unique(), 1, recent_block_hash);
        let new_last_entry = Entry::new(&prev_entry_hash, 1, vec![extra_tx]);
        let mut new_last_hash = new_last_entry.hash;

        // Now add back original second part, try to maintain same number of hashes in each tick.
        for entry in original_suffix {
            let mut num_hashes = entry.num_hashes;
            // We added one more new transaction above to make all duplicate blocks
            // different, but the number of hashes should still be the same, so we
            // are trying to adjust number of hashes in the first non-full tick to
            // accomodate the new transaction. A tick entry is basically empty hashes
            // patched after existing transactions, we are not touching the non-tick
            // entries because the tick could be already full, adding one more entry
            // would mean this tick has too many hashes and packet verification at
            // the receiving end would then fail.
            if should_decrease_hashes && entry.is_tick() {
                // When an entry is a tick it could not have zero hash.
                assert!(num_hashes > 0);
                num_hashes -= 1;
                should_decrease_hashes = false;
            }
            entry.update_hash_params(&new_last_hash, num_hashes);
            new_last_hash = entry.hash;
        }
        new_last_entry
    }

    fn send_all_shreds(
        &mut self,
        socket_sender: &Sender<(Arc<Vec<Shred>>, Option<BroadcastShredBatchInfo>)>,
        blockstore_sender: &Sender<(Arc<Vec<Shred>>, Option<BroadcastShredBatchInfo>)>,
        process_stats: &mut ProcessShredsStats,
    ) -> Result<()> {
        let mut coding_send_time = Measure::start("broadcast_coding_send");

        let mut result = Ok(());
        let now = Instant::now();
        self.shreds_to_send.retain(|x| {
            if now >= x.send_after {
                match send_shreds(x, socket_sender, blockstore_sender) {
                    Ok(_) => (),
                    e => result = e,
                }
                false
            } else {
                true
            }
        });
        coding_send_time.stop();

        process_stats.coding_send_elapsed = coding_send_time.as_us();

        result
    }

    fn insert(
        &mut self,
        blockstore: &Blockstore,
        shreds: Arc<Vec<Shred>>,
        broadcast_shred_batch_info: Option<BroadcastShredBatchInfo>,
    ) {
        // Insert shreds into blockstore
        let insert_shreds_start = Instant::now();
        // The first data shred is inserted synchronously.
        // https://github.com/solana-labs/solana/blob/92a0b310c/turbine/src/broadcast_stage/standard_broadcast_run.rs#L268-L283
        let offset = shreds
            .first()
            .map(|shred| shred.is_data() && shred.index() == 0)
            .map(usize::from)
            .unwrap_or_default();
        let num_shreds = shreds.len();
        let shreds = shreds.iter().skip(offset).map(Cow::Borrowed);
        blockstore
            .insert_cow_shreds(
                shreds, /*leader_schedule:*/ None, /*is_trusted:*/ true,
            )
            .expect("Failed to insert shreds in blockstore");
        let insert_shreds_elapsed = insert_shreds_start.elapsed();
        let new_insert_shreds_stats = InsertShredsStats {
            insert_shreds_elapsed: insert_shreds_elapsed.as_micros() as u64,
            num_shreds,
        };
        self.update_insertion_metrics(&new_insert_shreds_stats, &broadcast_shred_batch_info);
    }

    fn update_insertion_metrics(
        &mut self,
        new_insertion_shreds_stats: &InsertShredsStats,
        broadcast_shred_batch_info: &Option<BroadcastShredBatchInfo>,
    ) {
        let mut insert_shreds_stats = self.insert_shreds_stats.lock().unwrap();
        insert_shreds_stats.update(new_insertion_shreds_stats, broadcast_shred_batch_info);
    }

    fn broadcast(
        &mut self,
        sock: BroadcastSocket,
        cluster_info: &ClusterInfo,
        shreds: Arc<Vec<Shred>>,
        broadcast_shred_batch_info: &Option<BroadcastShredBatchInfo>,
        bank_forks: &RwLock<BankForks>,
        quic_endpoint_sender: &AsyncSender<(SocketAddr, Bytes)>,
    ) -> Result<()> {
        trace!("Broadcasting {:?} shreds", shreds.len());
        let mut transmit_stats = TransmitShredsStats {
            is_xdp: matches!(sock, BroadcastSocket::Xdp(_)),
            ..Default::default()
        };
        // Broadcast the shreds
        let mut transmit_time = Measure::start("broadcast_shreds");

        transmit_stats.num_shreds = shreds.len();
        let destinations = broadcast_shred_batch_info
            .as_ref()
            .and_then(|info| info.destinations.clone());
        broadcast_shreds(
            sock,
            &shreds,
            &self.cluster_nodes_cache,
            &self.last_datapoint_submit,
            &mut transmit_stats,
            cluster_info,
            bank_forks,
            cluster_info.socket_addr_space(),
            quic_endpoint_sender,
            destinations.as_ref().map(|d| d.as_slice()),
        )?;
        transmit_time.stop();

        transmit_stats.transmit_elapsed = transmit_time.as_us();

        // Process metrics
        self.update_transmit_metrics(&transmit_stats, broadcast_shred_batch_info);
        Ok(())
    }

    fn update_transmit_metrics(
        &mut self,
        new_transmit_shreds_stats: &TransmitShredsStats,
        broadcast_shred_batch_info: &Option<BroadcastShredBatchInfo>,
    ) {
        let mut transmit_shreds_stats = self.transmit_shreds_stats.lock().unwrap();
        transmit_shreds_stats.update(new_transmit_shreds_stats, broadcast_shred_batch_info);
    }

    fn report_and_reset_stats(&mut self, was_interrupted: bool) {
        let (name, slot_broadcast_time) = if was_interrupted {
            ("broadcast-process-shreds-interrupted-stats", None)
        } else {
            (
                "broadcast-process-shreds-stats",
                Some(self.slot_broadcast_start.elapsed()),
            )
        };

        self.process_shreds_stats.submit(
            name,
            self.slot,
            self.next_shred_index, // num_data_shreds
            self.next_code_index,  // num_coding_shreds
            slot_broadcast_time,
        );
    }
}

impl BroadcastRun for StandardBroadcastRun {
    fn run(
        &mut self,
        keypair: &Keypair,
        blockstore: &Blockstore,
        receiver: &Receiver<WorkingBankEntry>,
        socket_sender: &Sender<(Arc<Vec<Shred>>, Option<BroadcastShredBatchInfo>)>,
        blockstore_sender: &Sender<(Arc<Vec<Shred>>, Option<BroadcastShredBatchInfo>)>,
    ) -> Result<()> {
        if !self.shreds_to_send.is_empty() {
            let mut process_stats = ProcessShredsStats::default();
            self.send_all_shreds(socket_sender, blockstore_sender, &mut process_stats)?;
            self.process_shreds_stats += process_stats;
        }

        let mut process_stats = ProcessShredsStats::default();
        let receive_results = broadcast_utils::recv_slot_entries(
            receiver,
            &mut self.carryover_entry,
            &mut process_stats,
        )?;

        let adv_config =
            solana_adversary::adversary_feature_set::send_duplicate_blocks::get_config();

        // TODO: Confirm that last chunk of coding shreds
        // will not be lost or delayed for too long.
        self.process_receive_results(
            &adv_config,
            keypair,
            blockstore,
            socket_sender,
            blockstore_sender,
            receive_results,
            &mut process_stats,
        )?;
        self.send_all_shreds(socket_sender, blockstore_sender, &mut process_stats)?;
        self.process_shreds_stats += process_stats;
        Ok(())
    }
    fn transmit(
        &mut self,
        receiver: &TransmitReceiver,
        cluster_info: &ClusterInfo,
        sock: BroadcastSocket,
        bank_forks: &RwLock<BankForks>,
        quic_endpoint_sender: &AsyncSender<(SocketAddr, Bytes)>,
    ) -> Result<()> {
        let (shreds, batch_info) = receiver.recv()?;
        self.broadcast(
            sock,
            cluster_info,
            shreds,
            &batch_info,
            bank_forks,
            quic_endpoint_sender,
        )
    }
    fn record(&mut self, receiver: &RecordReceiver, blockstore: &Blockstore) -> Result<()> {
        let (shreds, slot_start_ts) = receiver.recv()?;
        self.insert(blockstore, shreds, slot_start_ts);
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use {
        super::*,
        rand::Rng,
        solana_entry::entry::create_ticks,
        solana_genesis_config::GenesisConfig,
        solana_gossip::{cluster_info::ClusterInfo, node::Node},
        solana_hash::Hash,
        solana_keypair::Keypair,
        solana_ledger::{
            blockstore::Blockstore,
            genesis_utils::create_genesis_config,
            get_tmp_ledger_path_auto_delete,
            shred::{max_ticks_per_n_shreds, DATA_SHREDS_PER_FEC_BLOCK},
        },
        solana_net_utils::sockets::bind_to_localhost_unique,
        solana_runtime::bank::Bank,
        solana_signer::Signer,
        solana_streamer::socket::SocketAddrSpace,
        std::{ops::Deref, sync::Arc, time::Duration},
    };

    #[allow(clippy::type_complexity)]
    fn setup(
        num_shreds_per_slot: Slot,
    ) -> (
        Arc<Blockstore>,
        GenesisConfig,
        Arc<ClusterInfo>,
        Arc<Bank>,
        Arc<Keypair>,
        UdpSocket,
        Arc<RwLock<BankForks>>,
    ) {
        // Setup
        let ledger_path = get_tmp_ledger_path_auto_delete!();
        let blockstore = Arc::new(
            Blockstore::open(ledger_path.path())
                .expect("Expected to be able to open database ledger"),
        );
        let leader_keypair = Arc::new(Keypair::new());
        let leader_pubkey = leader_keypair.pubkey();
        let leader_info = Node::new_localhost_with_pubkey(&leader_pubkey);
        let cluster_info = Arc::new(ClusterInfo::new(
            leader_info.info,
            leader_keypair.clone(),
            SocketAddrSpace::Unspecified,
        ));
        let socket = bind_to_localhost_unique().expect("should bind");
        let mut genesis_config = create_genesis_config(10_000).genesis_config;
        genesis_config.ticks_per_slot = max_ticks_per_n_shreds(num_shreds_per_slot, None) + 1;

        let bank = Bank::new_for_tests(&genesis_config);
        let bank_forks = BankForks::new_rw_arc(bank);
        let bank0 = bank_forks.read().unwrap().root_bank();
        (
            blockstore,
            genesis_config,
            cluster_info,
            bank0,
            leader_keypair,
            socket,
            bank_forks,
        )
    }

    #[test]
    fn test_interrupted_slot_last_shred() {
        let keypair = Arc::new(Keypair::new());
        let mut run = StandardBroadcastRun::new(0);
        assert!(run.completed);
        adversary_feature_set::send_duplicate_blocks::set_config(AdversarialConfig {
            num_duplicate_validators: 1,
            new_entry_index_from_end: 1,
            send_original_after_ms: 0,
            turbine_send_delay_ms: 0,
            send_destinations: vec![],
        });

        // Set up the slot to be interrupted
        let next_shred_index = 10;
        let slot = 1;
        let parent = 0;
        run.chained_merkle_root = Hash::new_from_array(rand::thread_rng().gen());
        run.next_shred_index = next_shred_index;
        run.next_code_index = 17;
        run.slot = slot;
        run.parent = parent;
        run.completed = false;
        run.slot_broadcast_start = Instant::now();

        // Slot 2 interrupted slot 1
        let shreds = run.finish_prev_slot(
            &keypair,
            0, // max_ticks_in_slot
            &mut ProcessShredsStats::default(),
        );
        assert!(run.completed);
        let shred = shreds
            .first()
            .expect("Expected a shred that signals an interrupt");

        // Validate the shred
        assert_eq!(shred.parent().unwrap(), parent);
        assert_eq!(shred.slot(), slot);
        assert_eq!(shred.index(), next_shred_index);
        assert!(shred.is_data());
        assert!(shred.verify(&keypair.pubkey()));
    }

    #[test]
    fn test_slot_interrupt() {
        // Setup
        let num_shreds_per_slot = DATA_SHREDS_PER_FEC_BLOCK as u64;
        let (blockstore, genesis_config, cluster_info, bank0, leader_keypair, socket, bank_forks) =
            setup(num_shreds_per_slot);
        let (quic_endpoint_sender, _quic_endpoint_receiver) =
            tokio::sync::mpsc::channel(/*capacity:*/ 128);

        // Insert 1 less than the number of ticks needed to finish the slot
        let ticks0 = create_ticks(genesis_config.ticks_per_slot - 1, 0, genesis_config.hash());
        let receive_results = ReceiveResults {
            entries: ticks0.clone(),
            bank: bank0.clone(),
            last_tick_height: (ticks0.len() - 1) as u64,
        };

        // Step 1: Make an incomplete transmission for slot 0
        let mut broadcast_duplicate_blocks_run = StandardBroadcastRun::new(0);
        let config = AdversarialConfig {
            num_duplicate_validators: 1,
            new_entry_index_from_end: 1,
            send_original_after_ms: 0,
            turbine_send_delay_ms: 0,
            send_destinations: vec![],
        };
        broadcast_duplicate_blocks_run
            .test_process_receive_results(
                &config,
                &leader_keypair,
                &cluster_info,
                &socket,
                &blockstore,
                receive_results,
                &bank_forks,
                &quic_endpoint_sender,
                1,
            )
            .unwrap();
        assert_eq!(
            broadcast_duplicate_blocks_run.next_shred_index as u64,
            num_shreds_per_slot
        );
        assert_eq!(broadcast_duplicate_blocks_run.slot, 0);
        assert_eq!(broadcast_duplicate_blocks_run.parent, 0);
        // Make sure the slot is not complete
        assert!(!blockstore.is_full(0));
        // Modify the stats, should reset later
        broadcast_duplicate_blocks_run
            .process_shreds_stats
            .receive_elapsed = 10;
        // Broadcast stats should exist, and 1 batch should have been sent,
        // for both data and coding shreds.
        //
        // But as we are duplicating the shreds, we actually see two batches being sent.
        assert_eq!(
            broadcast_duplicate_blocks_run
                .transmit_shreds_stats
                .lock()
                .unwrap()
                .get(broadcast_duplicate_blocks_run.slot)
                .unwrap()
                .num_batches(),
            2
        );
        assert_eq!(
            broadcast_duplicate_blocks_run
                .insert_shreds_stats
                .lock()
                .unwrap()
                .get(broadcast_duplicate_blocks_run.slot)
                .unwrap()
                .num_batches(),
            2
        );
        // Try to fetch ticks from blockstore, nothing should break
        assert_eq!(blockstore.get_slot_entries(0, 0).unwrap(), ticks0);
        assert_eq!(
            blockstore.get_slot_entries(0, num_shreds_per_slot).unwrap(),
            vec![],
        );

        // Step 2: Make a transmission for another bank that interrupts the transmission for
        // slot 0
        let bank2 = Arc::new(Bank::new_from_parent(bank0, &leader_keypair.pubkey(), 2));
        let interrupted_slot = broadcast_duplicate_blocks_run.slot;
        // Interrupting the slot should cause the unfinished_slot and stats to reset
        let num_shreds = 1;
        assert!(num_shreds < num_shreds_per_slot);
        let ticks1 = create_ticks(
            max_ticks_per_n_shreds(num_shreds, None),
            0,
            genesis_config.hash(),
        );
        let receive_results = ReceiveResults {
            entries: ticks1.clone(),
            bank: bank2,
            last_tick_height: (ticks1.len() - 1) as u64,
        };
        broadcast_duplicate_blocks_run
            .test_process_receive_results(
                &config,
                &leader_keypair,
                &cluster_info,
                &socket,
                &blockstore,
                receive_results,
                &bank_forks,
                &quic_endpoint_sender,
                1,
            )
            .unwrap();

        // The shred index should have reset to 0, which makes it possible for the
        // index < the previous shred index for slot 0
        assert_eq!(
            broadcast_duplicate_blocks_run.next_shred_index as usize,
            DATA_SHREDS_PER_FEC_BLOCK
        );
        assert_eq!(broadcast_duplicate_blocks_run.slot, 2);
        assert_eq!(broadcast_duplicate_blocks_run.parent, 0);

        // Check that the stats were reset as well
        assert_eq!(
            broadcast_duplicate_blocks_run
                .process_shreds_stats
                .receive_elapsed,
            0
        );

        // Broadcast stats for interrupted slot should be cleared
        assert!(broadcast_duplicate_blocks_run
            .transmit_shreds_stats
            .lock()
            .unwrap()
            .get(interrupted_slot)
            .is_none());
        assert!(broadcast_duplicate_blocks_run
            .insert_shreds_stats
            .lock()
            .unwrap()
            .get(interrupted_slot)
            .is_none());

        // Try to fetch the incomplete ticks from blockstore, should succeed
        assert_eq!(blockstore.get_slot_entries(0, 0).unwrap(), ticks0);
        assert_eq!(
            blockstore.get_slot_entries(0, num_shreds_per_slot).unwrap(),
            vec![],
        );
    }

    #[test]
    fn test_buffer_data_shreds() {
        let num_shreds_per_slot = 2;
        let (blockstore, genesis_config, _cluster_info, bank, leader_keypair, _socket, _bank_forks) =
            setup(num_shreds_per_slot);
        let (bsend, brecv) = unbounded();
        let (ssend, _srecv) = unbounded();
        let mut last_tick_height = 0;
        let mut broadcast_duplicate_blocks_run = StandardBroadcastRun::new(0);
        let mut process_stats = ProcessShredsStats::default();
        let mut process_ticks = |num_ticks| {
            let ticks = create_ticks(num_ticks, 0, genesis_config.hash());
            last_tick_height += (ticks.len() - 1) as u64;
            let receive_results = ReceiveResults {
                entries: ticks,
                bank: bank.clone(),
                last_tick_height,
            };
            broadcast_duplicate_blocks_run
                .process_receive_results(
                    &AdversarialConfig {
                        num_duplicate_validators: 1,
                        new_entry_index_from_end: 1,
                        send_original_after_ms: 0,
                        turbine_send_delay_ms: 0,
                        send_destinations: vec![],
                    },
                    &leader_keypair,
                    &blockstore,
                    &ssend,
                    &bsend,
                    receive_results,
                    &mut process_stats,
                )
                .unwrap();
            broadcast_duplicate_blocks_run
                .send_all_shreds(&ssend, &bsend, &mut process_stats)
                .unwrap();
        };
        for i in 0..3 {
            process_ticks((i + 1) * 100);
        }
        let mut shreds = Vec::<Shred>::new();
        while let Ok((recv_shreds, _)) = brecv.recv_timeout(Duration::from_secs(1)) {
            shreds.extend(recv_shreds.deref().clone());
        }
        // At least as many coding shreds as data shreds. The duplicates are not sent for now because we set a delay of half second
        assert!(shreds.len() >= DATA_SHREDS_PER_FEC_BLOCK * 2);
        assert_eq!(
            shreds.iter().filter(|shred| shred.is_data()).count(),
            shreds.len() / 2
        );
        process_ticks(75);
        while let Ok((recv_shreds, _)) = brecv.recv_timeout(Duration::from_secs(1)) {
            shreds.extend(recv_shreds.deref().clone());
        }
        assert!(shreds.len() >= DATA_SHREDS_PER_FEC_BLOCK * 2);
        assert_eq!(
            shreds.iter().filter(|shred| shred.is_data()).count(),
            shreds.len() / 2
        );
    }

    #[test]
    fn test_slot_finish() {
        // Setup
        let num_shreds_per_slot = 2;
        let (blockstore, genesis_config, cluster_info, bank0, leader_keypair, socket, bank_forks) =
            setup(num_shreds_per_slot);
        let (quic_endpoint_sender, _quic_endpoint_receiver) =
            tokio::sync::mpsc::channel(/*capacity:*/ 128);

        // Insert complete slot of ticks needed to finish the slot
        let ticks = create_ticks(genesis_config.ticks_per_slot, 0, genesis_config.hash());
        let receive_results = ReceiveResults {
            entries: ticks.clone(),
            bank: bank0,
            last_tick_height: ticks.len() as u64,
        };

        let mut broadcast_duplicate_blocks_run = StandardBroadcastRun::new(0);
        broadcast_duplicate_blocks_run
            .test_process_receive_results(
                &AdversarialConfig {
                    num_duplicate_validators: 1,
                    new_entry_index_from_end: 1,
                    send_original_after_ms: 0,
                    turbine_send_delay_ms: 0,
                    send_destinations: vec![],
                },
                &leader_keypair,
                &cluster_info,
                &socket,
                &blockstore,
                receive_results,
                &bank_forks,
                &quic_endpoint_sender,
                1,
            )
            .unwrap();
        assert!(broadcast_duplicate_blocks_run.completed)
    }

    #[cfg(test)]
    fn verify_chain_of_hashes(new_entries: &Vec<Entry>) {
        let mut last_hash = Hash::default();
        for entry in new_entries {
            assert!(entry.verify(&last_hash));
            last_hash = entry.hash;
        }
    }

    #[test]
    fn test_duplicate_blocks_generation() {
        // Setup
        let keypair = Arc::new(Keypair::new());

        // Test the case where hashes_per_tick is 0, we can insert additional transactions freely in
        // this case.
        let mut hashes_per_tick: usize = 0;
        let mut num_ticks: usize = 5;
        let mut ticks = create_ticks(
            num_ticks.try_into().unwrap(),
            hashes_per_tick.try_into().unwrap(),
            Hash::default(),
        );
        let mut last_hash = Hash::default();
        let mut entries_kept: usize = num_ticks - 1;

        // Calculate duplicate blocks and insert one transaction before the last entry.
        let mut popped = ticks.split_off(entries_kept);
        ticks.push(StandardBroadcastRun::calculate_duplicate_blocks_suffix(
            &mut popped,
            false,
            ticks.last().unwrap().hash,
            Hash::default(),
            keypair.as_ref(),
        ));
        ticks.extend(popped);
        let mut expected_hashes: Vec<u64> =
            vec![hashes_per_tick.try_into().unwrap(); num_ticks + 1];
        expected_hashes[entries_kept] = 1;
        let actual_hashes: Vec<u64> = ticks.iter().map(|x| x.num_hashes).collect();
        assert_eq!(
            actual_hashes[..],
            expected_hashes[..],
            "\nExpected\n{:?}\nfound\n{:?}",
            &expected_hashes[..],
            &actual_hashes[..]
        );
        verify_chain_of_hashes(&ticks);
        assert_ne!(last_hash, ticks.last().unwrap().hash);
        last_hash = ticks.last().unwrap().hash;

        // Calculate for next iteration, no more insert, but hash is different.
        let mut popped = ticks.split_off(entries_kept);
        popped.remove(0);
        ticks.push(StandardBroadcastRun::calculate_duplicate_blocks_suffix(
            &mut popped,
            false,
            ticks.last().unwrap().hash,
            Hash::default(),
            keypair.as_ref(),
        ));
        ticks.extend(popped);
        let actual_hashes: Vec<u64> = ticks.iter().map(|x| x.num_hashes).collect();
        assert_eq!(
            actual_hashes[..],
            expected_hashes[..],
            "\nExpected\n{:?}\nfound\n{:?}",
            &expected_hashes[..],
            &actual_hashes[..]
        );
        verify_chain_of_hashes(&ticks);
        assert_ne!(last_hash, ticks.last().unwrap().hash);

        // Now test the case where hashes_per_tick is non-zero, we do have to pack the extra
        // transaction into existing ticks, so the first tick after insertion will have its
        // num_hashes decreased.
        hashes_per_tick = 10;
        num_ticks = 6;
        entries_kept = num_ticks - 2;
        ticks = create_ticks(
            num_ticks.try_into().unwrap(),
            hashes_per_tick.try_into().unwrap(),
            Hash::default(),
        );

        // Calculate duplicate blocks and insert one transaction before the last two entries.
        let mut popped = ticks.split_off(entries_kept);
        ticks.push(StandardBroadcastRun::calculate_duplicate_blocks_suffix(
            &mut popped,
            true,
            ticks.last().unwrap().hash,
            Hash::default(),
            keypair.as_ref(),
        ));
        ticks.extend(popped);
        expected_hashes = vec![hashes_per_tick.try_into().unwrap(); num_ticks + 1];
        expected_hashes[entries_kept] = 1;
        expected_hashes[entries_kept + 1] = (hashes_per_tick - 1).try_into().unwrap();
        let actual_hashes: Vec<u64> = ticks.iter().map(|x| x.num_hashes).collect();
        assert_eq!(
            actual_hashes[..],
            expected_hashes[..],
            "\nExpected\n{:?}\nfound\n{:?}",
            &expected_hashes[..],
            &actual_hashes[..]
        );
        verify_chain_of_hashes(&ticks);
        assert_ne!(last_hash, ticks.last().unwrap().hash);
        last_hash = ticks.last().unwrap().hash;

        // Calculate for next iteration, no more insert, but hash is different
        let mut popped = ticks.split_off(entries_kept);
        popped.remove(0);
        ticks.push(StandardBroadcastRun::calculate_duplicate_blocks_suffix(
            &mut popped,
            false,
            ticks.last().unwrap().hash,
            Hash::default(),
            keypair.as_ref(),
        ));
        ticks.extend(popped);
        let actual_hashes: Vec<u64> = ticks.iter().map(|x| x.num_hashes).collect();
        assert_eq!(
            actual_hashes[..],
            expected_hashes[..],
            "\nExpected\n{:?}\nfound\n{:?}",
            &expected_hashes[..],
            &actual_hashes[..]
        );
        verify_chain_of_hashes(&ticks);
        assert_ne!(last_hash, ticks.last().unwrap().hash);
    }

    #[test]
    fn test_send_duplicate_blocks_out() {
        // Setup
        let num_shreds_per_slot = 2;
        let (blockstore, genesis_config, cluster_info, bank0, leader_keypair, socket, bank_forks) =
            setup(num_shreds_per_slot);
        let (quic_endpoint_sender, _quic_endpoint_receiver) =
            tokio::sync::mpsc::channel(/*capacity:*/ 128);

        // Insert complete slot of ticks needed to finish the slot
        let tx0 = system_transaction::transfer(
            &leader_keypair,
            &Pubkey::new_unique(),
            1,
            Hash::default(),
        );
        let entry0 = Entry::new(&genesis_config.hash(), 1, vec![tx0]);
        let mut ticks0 = create_ticks(genesis_config.ticks_per_slot, 0, entry0.hash);
        ticks0.insert(0, entry0);
        let receive_results0 = ReceiveResults {
            entries: ticks0.clone(),
            bank: bank0.clone(),
            last_tick_height: (ticks0.len() - 1) as u64,
        };

        let mut broadcast_duplicate_blocks_run = StandardBroadcastRun::new(0);
        let config = AdversarialConfig {
            num_duplicate_validators: 2,
            new_entry_index_from_end: 1,
            send_original_after_ms: 0,
            turbine_send_delay_ms: 0,
            send_destinations: vec![],
        };
        broadcast_duplicate_blocks_run
            .test_process_receive_results(
                &config,
                &leader_keypair,
                &cluster_info,
                &socket,
                &blockstore,
                receive_results0,
                &bank_forks,
                &quic_endpoint_sender,
                3,
            )
            .unwrap();

        let tx1 = system_transaction::transfer(
            &leader_keypair,
            &Pubkey::new_unique(),
            1,
            Hash::default(),
        );
        let entry1 = Entry::new(&ticks0.last().unwrap().hash, 1, vec![tx1]);
        let bank1 = Arc::new(Bank::new_from_parent(bank0, &leader_keypair.pubkey(), 1));
        let receive_results1 = ReceiveResults {
            entries: vec![entry1.clone()],
            bank: bank1.clone(),
            last_tick_height: ticks0.len() as u64,
        };
        broadcast_duplicate_blocks_run
            .test_process_receive_results(
                &config,
                &leader_keypair,
                &cluster_info,
                &socket,
                &blockstore,
                receive_results1,
                &bank_forks,
                &quic_endpoint_sender,
                1,
            )
            .unwrap();
        let ticks1 = create_ticks(genesis_config.ticks_per_slot, 0, entry1.hash);
        let receive_results2 = ReceiveResults {
            entries: ticks1,
            bank: bank1,
            last_tick_height: (ticks0.len() * 2 - 2) as u64,
        };
        broadcast_duplicate_blocks_run
            .test_process_receive_results(
                &config,
                &leader_keypair,
                &cluster_info,
                &socket,
                &blockstore,
                receive_results2,
                &bank_forks,
                &quic_endpoint_sender,
                3,
            )
            .unwrap();
    }

    #[test]
    fn test_delay_first_block_broadcast_out() {
        // Setup
        let turbine_send_delay_ms = 1600;
        let num_shreds_per_slot = 2;
        let (blockstore, genesis_config, _, bank0, leader_keypair, _, _) =
            setup(num_shreds_per_slot);

        // Insert complete slot of ticks needed to finish the slot
        let tx0 = system_transaction::transfer(
            &leader_keypair,
            &Pubkey::new_unique(),
            1,
            Hash::default(),
        );
        let entry0 = Entry::new(&genesis_config.hash(), 1, vec![tx0]);
        let mut ticks0 = create_ticks(genesis_config.ticks_per_slot, 0, entry0.hash);
        ticks0.insert(0, entry0);
        let receive_results0 = ReceiveResults {
            entries: ticks0.clone(),
            bank: bank0.clone(),
            last_tick_height: (ticks0.len() - 1) as u64,
        };

        let mut broadcast_run = StandardBroadcastRun::new(0);
        let config = AdversarialConfig {
            num_duplicate_validators: 0,
            new_entry_index_from_end: 0,
            send_original_after_ms: 0,
            turbine_send_delay_ms,
            send_destinations: vec![],
        };
        let (bsend, _) = unbounded();
        let (ssend, _) = unbounded();
        let mut process_stats = ProcessShredsStats::default();
        let shred_send_minimum = Instant::now() + Duration::from_millis(turbine_send_delay_ms);
        broadcast_run
            .process_receive_results(
                &config,
                &leader_keypair,
                &blockstore,
                &ssend,
                &bsend,
                receive_results0,
                &mut process_stats,
            )
            .expect("process_receive_results failed");

        assert!(!broadcast_run.shreds_to_send.is_empty());
        for shred in broadcast_run.shreds_to_send.iter() {
            assert!(
                shred.send_after >= shred_send_minimum,
                "All produced shreds must have `send_after` set at least `turbine_send_delay_ms` \
                 ({}) ms in the future.\nGot a shred with `send_after`: {:?}\nExpected minimum: \
                 {:?}",
                turbine_send_delay_ms,
                shred.send_after,
                shred_send_minimum
            );
        }
    }
}
