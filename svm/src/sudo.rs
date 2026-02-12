use {
    crate::{
        account_loader::LoadedTransaction,
        message_processor::process_message,
        nonce_info::NonceInfo,
        rollback_accounts::RollbackAccounts,
        transaction_error_metrics::TransactionErrorMetrics,
        transaction_execution_result::{ExecutedTransaction, TransactionExecutionDetails},
        transaction_processor::{
            TransactionLogMessages, TransactionProcessingConfig, TransactionProcessingEnvironment,
        },
    },
    agave_transaction_view::{
        resolved_transaction_view::ResolvedTransactionView, transaction_data::TransactionData,
        transaction_version::TransactionVersion, transaction_view::TransactionView,
    },
    solana_account::{AccountSharedData, ReadableAccount, WritableAccount, state_traits::StateMut},
    solana_address_lookup_table_interface::state::AddressLookupTable,
    solana_clock::Slot,
    solana_fee_structure::FeeDetails,
    solana_instruction::{TRANSACTION_LEVEL_STACK_HEIGHT, error::InstructionError},
    solana_message::{
        AccountKeys,
        compiled_instruction::CompiledInstruction,
        inner_instruction::{InnerInstruction, InnerInstructionsList},
        v0::LoadedAddresses,
    },
    solana_nonce::{
        NONCED_TX_MARKER_IX_INDEX,
        state::{DurableNonce, State as NonceState},
        versions::Versions as NonceVersions,
    },
    solana_nonce_account::verify_nonce_account,
    solana_program_runtime::{
        execution_budget::{
            DEFAULT_HEAP_COST, SVMTransactionExecutionBudget, SVMTransactionExecutionCost,
        },
        invoke_context::{EnvironmentConfig, InvokeContext},
        loaded_programs::ProgramCacheForTxBatch,
        sysvar_cache::SysvarCache,
        vm::calculate_heap_cost,
    },
    solana_pubkey::{Pubkey, pubkey},
    solana_svm_callback::TransactionProcessingCallback,
    solana_svm_log_collector::LogCollector,
    solana_svm_timings::ExecuteTimings,
    solana_svm_transaction::{svm_message::SVMMessage, svm_transaction::SVMTransaction},
    solana_transaction::{TransactionResult, sanitized::MAX_TX_ACCOUNT_LOCKS},
    solana_transaction_context::{
        IndexOfAccount,
        transaction::{ExecutionRecord, TransactionContext},
    },
    solana_transaction_error::TransactionError,
    std::{
        cell::RefCell,
        collections::{HashMap, HashSet},
        rc::Rc,
        sync::RwLock,
    },
};

const SUDO_PROGRAM_ID: Pubkey = pubkey!("Sudo111111111111111111111111111111111111111");
const BASE_SUDO_COST: u64 = 50_000;

pub fn is_sudo_tx(tx: &impl SVMTransaction) -> bool {
    tx.program_instructions_iter()
        .next()
        .is_some_and(|(pid, _)| pid == &SUDO_PROGRAM_ID)
}

#[allow(clippy::too_many_arguments)]
pub fn execute_loaded_transaction<CB: TransactionProcessingCallback>(
    sysvar_cache: &RwLock<SysvarCache>,
    execution_cost: SVMTransactionExecutionCost,
    callback: &CB,
    tx: &impl SVMTransaction,
    mut loaded_transaction: LoadedTransaction,
    execute_timings: &mut ExecuteTimings,
    error_metrics: &mut TransactionErrorMetrics,
    program_cache_for_tx_batch: &mut ProgramCacheForTxBatch,
    environment: &TransactionProcessingEnvironment,
    config: &TransactionProcessingConfig,
    current_slot: Slot,
) -> ExecutedTransaction {
    match execute_loaded_transaction_inner(
        sysvar_cache,
        execution_cost,
        callback,
        tx,
        &mut loaded_transaction,
        execute_timings,
        error_metrics,
        program_cache_for_tx_batch,
        environment,
        config,
        current_slot,
    ) {
        Some(execution_details) => ExecutedTransaction {
            loaded_transaction,
            execution_details,
            programs_modified_by_tx: program_cache_for_tx_batch.drain_modified_entries(),
        },
        None => ExecutedTransaction {
            loaded_transaction,
            execution_details: TransactionExecutionDetails {
                status: Err(TransactionError::InstructionError(
                    0,
                    InstructionError::ArithmeticOverflow,
                )),
                log_messages: None,
                inner_instructions: None,
                return_data: None,
                executed_units: BASE_SUDO_COST,
                accounts_data_len_delta: 0,
            },
            programs_modified_by_tx: HashMap::new(),
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_loaded_transaction_inner<CB: TransactionProcessingCallback>(
    sysvar_cache: &RwLock<SysvarCache>,
    execution_cost: SVMTransactionExecutionCost,
    callback: &CB,
    tx: &impl SVMTransaction,
    loaded_transaction: &mut LoadedTransaction,
    execute_timings: &mut ExecuteTimings,
    error_metrics: &mut TransactionErrorMetrics,
    program_cache_for_tx_batch: &mut ProgramCacheForTxBatch,
    environment: &TransactionProcessingEnvironment,
    config: &TransactionProcessingConfig,
    current_slot: Slot,
) -> Option<TransactionExecutionDetails> {
    // Extract the sudo instruction.
    let sudo_ix = tx.instructions_iter().next()?;
    let sudo_ix = SudoInstructionData::deserialize(sudo_ix.data)?;

    // Deserialize & sanitize the inner transaction.
    //
    // NOTE: We use an empty reserved_account_keys set because the outer TX
    // already validated that reserved keys aren't locked.
    let reserved_account_keys = HashSet::new();
    let inner_data = sudo_ix.inner_tx_bytes.as_slice();
    let (inner, alt_deactivation_slot) = translate_to_resolved_view(
        inner_data,
        &loaded_transaction.accounts,
        &sudo_ix.account_map,
        &reserved_account_keys,
    )?;

    // Verify ALT deactivation slot.
    if current_slot >= alt_deactivation_slot {
        return None;
    }

    // Verify inner TX signatures.
    for (i, signature) in inner.signatures().iter().enumerate() {
        let signer_pubkey = inner.static_account_keys().get(i)?;
        if !signature.verify(signer_pubkey.as_ref(), inner.message_data()) {
            return None;
        }
    }

    // Verify inner <> outer key mapping is consistent.
    let inner_keys = inner.account_keys();
    if sudo_ix.account_map.len() != inner_keys.len() {
        return None;
    }
    for (inner_idx, &outer_idx) in sudo_ix.account_map.iter().enumerate() {
        let inner_key = inner_keys.get(inner_idx)?;
        let (outer_key, _) = loaded_transaction.accounts.get(outer_idx as usize)?;
        if inner_key != outer_key {
            return None;
        }
    }

    // Parse inner TX's compute budget instructions.
    let inner_limits = parse_inner_compute_budget(&inner)?;

    // Verify outer TX has sufficient compute budget for inner TX execution.
    let heap_cost = calculate_heap_cost(inner_limits.heap_size, DEFAULT_HEAP_COST);
    let required_cu = BASE_SUDO_COST
        .checked_add(inner_limits.compute_unit_limit)?
        .checked_add(heap_cost)?;
    if loaded_transaction.compute_budget.compute_unit_limit < required_cu {
        return None;
    }

    // Verify replay protection and advance nonce (if applicable).
    match inner.get_durable_nonce(true) {
        Some(nonce_address) => {
            let nonce_info = validate_nonce(
                &inner,
                nonce_address,
                &inner_keys,
                &sudo_ix.account_map,
                &loaded_transaction.accounts,
                environment,
            )?;

            // Write advanced nonce back to outer accounts.
            let nonce_inner_idx = inner_keys.iter().position(|k| k == nonce_address)?;
            let nonce_outer_idx = *sudo_ix.account_map.get(nonce_inner_idx)? as usize;
            loaded_transaction.accounts[nonce_outer_idx].1 = nonce_info.account().clone();
        }
        None => unimplemented!("verify tombstone pda"),
    };

    // Calculate and transfer inner TX fees.
    transfer_inner_fee(
        calculate_inner_fee(&inner, &inner_limits, environment),
        &sudo_ix.account_map,
        &mut loaded_transaction.accounts,
    )?;

    // Build inner LoadedTransaction.
    let inner_loaded = build_inner_loaded_transaction(
        &inner,
        &inner_limits,
        &sudo_ix.account_map,
        &loaded_transaction.accounts,
        environment,
    )?;

    // Execute inner message.
    let inner_execution = execute_inner_message(
        &inner,
        inner_loaded,
        &sudo_ix.account_map,
        callback,
        program_cache_for_tx_batch,
        &sysvar_cache.read().unwrap(),
        environment,
        execution_cost,
        execute_timings,
        config,
    );

    // On success, sync all inner account changes back to outer accounts. On failure we
    // don't do anything as we've already synced nonce & fee payer.
    if inner_execution.status.is_ok() {
        for (inner_idx, &outer_idx) in sudo_ix.account_map.iter().enumerate() {
            loaded_transaction.accounts[outer_idx as usize].1 =
                inner_execution.accounts[inner_idx].1.clone();
        }
    }

    Some(TransactionExecutionDetails {
        status: Ok(()), // Sudo always succeeds if we got here
        log_messages: build_sudo_logs(inner_execution.log_messages, &inner_execution.status),
        inner_instructions: build_sudo_inner_instructions(
            inner_execution.inner_instructions,
            &sudo_ix.account_map,
        ),
        return_data: inner_execution.return_data,
        executed_units: BASE_SUDO_COST
            .checked_add(inner_execution.executed_units)
            .unwrap(),
        accounts_data_len_delta: inner_execution.accounts_data_len_delta,
    })
}

struct SudoInstructionData {
    /// Raw serialized inner transaction (legacy or V0).
    inner_tx_bytes: Vec<u8>,

    /// Maps inner account index -> outer account index.
    ///
    /// `outer_accounts[account_map[i]] == inner_accounts[i]`
    account_map: Vec<u8>,
}

impl SudoInstructionData {
    fn deserialize(_buf: &[u8]) -> Option<Self> {
        unimplemented!()
    }
}

/// Parses inner TX bytes into a resolved transaction view.
///
/// Uses the outer TX's accounts to resolve ALT addresses instead of loading from bank.
fn translate_to_resolved_view<D: TransactionData>(
    inner_tx_data: D,
    outer_accounts: &[(Pubkey, AccountSharedData)],
    account_map: &[u8],
    reserved_account_keys: &HashSet<Pubkey>,
) -> Option<(ResolvedTransactionView<D>, Slot)> {
    // Parsing and basic sanitization checks.
    let view = TransactionView::try_new_sanitized(inner_tx_data, true, true).ok()?;

    // TODO: Is `MAX_TX_ACCOUNT_LOCKS` even live?
    if usize::from(view.total_num_accounts()) > MAX_TX_ACCOUNT_LOCKS {
        return None;
    }

    // Resolve addresses using outer TX's accounts instead of loading from bank.
    let (loaded_addresses, deactivation_slot) =
        resolve_inner_addresses(&view, outer_accounts, account_map)?;

    let view =
        ResolvedTransactionView::try_new(view, loaded_addresses, reserved_account_keys).ok()?;

    // Validate no duplicate accounts (must be after resolution to catch ALT duplicates).
    validate_account_locks(view.account_keys(), MAX_TX_ACCOUNT_LOCKS).ok()?;

    Some((view, deactivation_slot))
}

/// Resolves inner TX addresses using the outer TX's already-loaded accounts.
///
/// For V0 transactions, the inner TX references ALTs which were already resolved
/// by the outer TX. We extract the resolved addresses from the outer TX's account
/// list using the account_map, and read deactivation_slot from the ALT accounts.
fn resolve_inner_addresses<D: TransactionData>(
    inner_view: &TransactionView<true, D>,
    outer_accounts: &[(Pubkey, AccountSharedData)],
    account_map: &[u8],
) -> Option<(Option<LoadedAddresses>, Slot)> {
    match inner_view.version() {
        TransactionVersion::Legacy => Some((None, Slot::MAX)),
        TransactionVersion::V0 => {
            let num_static = inner_view.static_account_keys().len();
            let num_writable_lookup = inner_view.total_writable_lookup_accounts() as usize;
            let num_readonly_lookup = inner_view.total_readonly_lookup_accounts() as usize;

            // Account map layout: [static keys...][writable lookup...][readonly lookup...]
            let writable_start = num_static;
            let readonly_start = writable_start + num_writable_lookup;
            let expected_len = readonly_start + num_readonly_lookup;
            if account_map.len() != expected_len {
                return None;
            }

            // Extract resolved writable addresses from outer accounts.
            let writable: Vec<Pubkey> = account_map[writable_start..readonly_start]
                .iter()
                .map(|&outer_idx| outer_accounts.get(outer_idx as usize).map(|(k, _)| *k))
                .collect::<Option<Vec<_>>>()?;

            // Extract resolved readonly addresses from outer accounts.
            let readonly: Vec<Pubkey> = account_map[readonly_start..]
                .iter()
                .map(|&outer_idx| outer_accounts.get(outer_idx as usize).map(|(k, _)| *k))
                .collect::<Option<Vec<_>>>()?;

            // Find minimum deactivation slot across all referenced ALTs.
            let mut deactivation_slot = Slot::MAX;
            for alt_lookup in inner_view.address_table_lookup_iter() {
                let alt_pubkey = alt_lookup.account_key;
                // TODO: Bit slow/risky.
                let (_, alt_account) = outer_accounts.iter().find(|(k, _)| k == alt_pubkey)?;

                let lookup_table = AddressLookupTable::deserialize(alt_account.data()).ok()?;
                deactivation_slot =
                    core::cmp::min(deactivation_slot, lookup_table.meta.deactivation_slot);
            }

            Some((
                Some(LoadedAddresses { writable, readonly }),
                deactivation_slot,
            ))
        }
    }
}

fn validate_account_locks(
    account_keys: AccountKeys,
    tx_account_lock_limit: usize,
) -> TransactionResult<()> {
    if account_keys.len() > tx_account_lock_limit {
        Err(TransactionError::TooManyAccountLocks)
    } else if has_duplicates(&account_keys) {
        Err(TransactionError::AccountLoadedTwice)
    } else {
        Ok(())
    }
}

fn has_duplicates(account_keys: &AccountKeys) -> bool {
    let mut seen = HashSet::with_capacity(account_keys.len());
    for key in account_keys.iter() {
        if !seen.insert(key) {
            return true;
        }
    }
    false
}

fn validate_nonce(
    inner: &impl SVMMessage,
    nonce_address: &Pubkey,
    inner_keys: &AccountKeys,
    account_map: &[u8],
    outer_accounts: &[(Pubkey, AccountSharedData)],
    environment: &TransactionProcessingEnvironment,
) -> Option<NonceInfo> {
    // Find the nonce account in the inner TX's account list.
    let nonce_inner_idx = inner_keys.iter().position(|k| k == nonce_address)?;
    let nonce_outer_idx = *account_map.get(nonce_inner_idx)? as usize;
    let (_, nonce_account) = outer_accounts.get(nonce_outer_idx)?;
    let mut nonce_account = nonce_account.clone();

    // Verify nonce account:
    // - Owner is SystemProgram
    // - State is Initialized
    // - Stored durable nonce matches inner TX's recent_blockhash
    let nonce_data = verify_nonce_account(&nonce_account, inner.recent_blockhash())?;

    // Verify nonce hasn't already been used this slot.
    let next_durable_nonce = DurableNonce::from_blockhash(&environment.blockhash);
    if nonce_data.durable_nonce == next_durable_nonce {
        return None; // Already used
    }

    // Verify nonce authority signed the advance nonce instruction (ix 0).
    let authority_signed = inner
        .get_ix_signers(NONCED_TX_MARKER_IX_INDEX as usize)
        .any(|signer| signer == &nonce_data.authority);
    if !authority_signed {
        return None;
    }

    // Advance the nonce to prevent replay.
    let next_nonce_state = NonceState::new_initialized(
        &nonce_data.authority,
        next_durable_nonce,
        environment.blockhash_lamports_per_signature,
    );
    nonce_account
        .set_state(&NonceVersions::new(next_nonce_state))
        .expect("Serializing into a validated nonce account cannot fail");

    Some(NonceInfo::new(*nonce_address, nonce_account))
}

fn calculate_inner_fee(
    inner: &impl SVMMessage,
    inner_limits: &InnerComputeBudgetLimits,
    environment: &TransactionProcessingEnvironment,
) -> FeeDetails {
    // Count all signatures (transaction + precompiles).
    let signature_count = inner
        .num_transaction_signatures()
        .saturating_add(inner.num_ed25519_signatures())
        .saturating_add(inner.num_secp256k1_signatures())
        .saturating_add(inner.num_secp256r1_signatures());

    let signature_fee =
        signature_count.saturating_mul(environment.blockhash_lamports_per_signature);

    let prioritization_fee = inner_limits
        .compute_unit_price
        .saturating_mul(inner_limits.compute_unit_limit)
        .saturating_add(999_999)
        .saturating_div(1_000_000);

    FeeDetails::new(signature_fee, prioritization_fee)
}

fn transfer_inner_fee(
    fee: FeeDetails,
    account_map: &[u8],
    loaded_accounts: &mut [(Pubkey, AccountSharedData)],
) -> Option<()> {
    let total_fee = fee.total_fee();
    if total_fee == 0 {
        return Some(());
    }

    // Resolve the inner fee payer to the account idx in the loaded_accounts.
    let inner_fee_payer_outer_idx = *account_map.first()?;
    let inner_fee_payer_outer_idx = inner_fee_payer_outer_idx as usize;

    // Deduct fee from inner fee payer.
    let (_, inner_payer_account) = loaded_accounts.get_mut(inner_fee_payer_outer_idx)?;
    inner_payer_account.checked_sub_lamports(total_fee).ok()?;

    // Credit fee to outer fee payer.
    let (_, outer_payer_account) = loaded_accounts.get_mut(0)?;
    outer_payer_account.checked_add_lamports(total_fee).ok()?;

    Some(())
}

fn build_inner_loaded_transaction(
    inner: &impl SVMMessage,
    inner_limits: &InnerComputeBudgetLimits,
    account_map: &[u8],
    outer_accounts: &[(Pubkey, AccountSharedData)],
    environment: &TransactionProcessingEnvironment,
) -> Option<LoadedTransaction> {
    // Map accounts.
    let inner_accounts: Vec<(Pubkey, AccountSharedData)> = account_map
        .iter()
        .map(|&outer_idx| outer_accounts.get(outer_idx as usize).cloned())
        .collect::<Option<Vec<_>>>()?;

    // Build program_indices.
    let program_indices: Vec<IndexOfAccount> = inner
        .instructions_iter()
        .map(|ix| ix.program_id_index as IndexOfAccount)
        .collect();

    // Build compute budget from parsed inner limits.
    let simd_0268_active = environment.feature_set.raise_cpi_nesting_limit_to_8;
    let compute_budget = SVMTransactionExecutionBudget {
        compute_unit_limit: inner_limits.compute_unit_limit,
        heap_size: inner_limits.heap_size,
        ..SVMTransactionExecutionBudget::new_with_defaults(simd_0268_active)
    };

    // Rollback is handled at the outer TX level (nonce already advanced, fees transferred).
    let rollback_accounts = RollbackAccounts::FeePayerOnly {
        fee_payer: inner_accounts.first()?.clone(),
    };

    Some(LoadedTransaction {
        accounts: inner_accounts,
        program_indices,
        // Already transferred to outer payer.
        fee_details: FeeDetails::default(),
        rollback_accounts,
        compute_budget,
        // Already validated via outer TX.
        loaded_accounts_data_size: 0,
    })
}

struct InnerComputeBudgetLimits {
    compute_unit_limit: u64,
    heap_size: u32,
    compute_unit_price: u64,
}

fn parse_inner_compute_budget(inner: &impl SVMMessage) -> Option<InnerComputeBudgetLimits> {
    unimplemented!()
}

struct InnerExecutionResult {
    status: Result<(), TransactionError>,
    accounts: Vec<(Pubkey, AccountSharedData)>,
    return_data: Option<solana_transaction_context::transaction::TransactionReturnData>,
    executed_units: u64,
    accounts_data_len_delta: i64,
    log_messages: Option<TransactionLogMessages>,
    inner_instructions: Option<InnerInstructionsList>,
}

#[allow(clippy::too_many_arguments)]
fn execute_inner_message<CB: TransactionProcessingCallback>(
    inner: &impl SVMMessage,
    inner_loaded: LoadedTransaction,
    account_map: &[u8],
    callback: &CB,
    program_cache_for_tx_batch: &mut ProgramCacheForTxBatch,
    sysvar_cache: &SysvarCache,
    environment: &TransactionProcessingEnvironment,
    execution_cost: SVMTransactionExecutionCost,
    execute_timings: &mut ExecuteTimings,
    config: &TransactionProcessingConfig,
) -> InnerExecutionResult {
    let compute_budget = inner_loaded.compute_budget;

    // Create transaction context with inner accounts.
    let mut transaction_context = TransactionContext::new(
        inner_loaded.accounts,
        environment.rent.clone(),
        compute_budget.max_instruction_stack_depth,
        compute_budget.max_instruction_trace_length,
        inner.num_instructions(),
    );

    // Create log collector if recording enabled.
    let log_collector = config.recording_config.enable_log_recording.then(|| {
        match config.log_messages_bytes_limit {
            None => LogCollector::new_ref(),
            Some(limit) => LogCollector::new_ref_with_limit(Some(limit)),
        }
    });

    // Create invoke context for inner execution.
    let mut executed_units = 0u64;
    let mut invoke_context = InvokeContext::new(
        &mut transaction_context,
        program_cache_for_tx_batch,
        EnvironmentConfig::new(
            environment.blockhash,
            environment.blockhash_lamports_per_signature,
            callback,
            &environment.feature_set,
            &environment.program_runtime_environments_for_execution,
            &environment.program_runtime_environments_for_deployment,
            sysvar_cache,
        ),
        log_collector.clone(),
        compute_budget,
        execution_cost,
    );

    // Execute the inner message.
    let status = process_message(
        inner,
        &inner_loaded.program_indices,
        &mut invoke_context,
        execute_timings,
        &mut executed_units,
    )
    .map(|_| ());

    // Must drop invoke_context before extracting from transaction_context.
    drop(invoke_context);

    // Extract log messages.
    let log_messages = log_collector.and_then(|lc| {
        Rc::try_unwrap(lc)
            .map(|lc| lc.into_inner().into_messages())
            .ok()
    });

    // Extract inner instructions if recording enabled.
    let inner_instructions = config
        .recording_config
        .enable_cpi_recording
        .then(|| extract_inner_instructions(&mut transaction_context, account_map));

    // Extract execution results.
    let ExecutionRecord {
        accounts,
        return_data,
        touched_account_count: _,
        accounts_resize_delta,
    } = transaction_context.into();
    let return_data = match return_data.data.is_empty() {
        true => None,
        false => Some(return_data),
    };

    InnerExecutionResult {
        status,
        accounts,
        return_data,
        executed_units,
        accounts_data_len_delta: accounts_resize_delta,
        log_messages,
        inner_instructions,
    }
}

fn extract_inner_instructions(
    transaction_context: &mut TransactionContext,
    account_map: &[u8],
) -> InnerInstructionsList {
    let (ix_trace, accounts, ix_data_trace) = transaction_context.take_instruction_trace();

    let mut result: Vec<InnerInstruction> = Vec::new();

    for ((ix_frame, ix_data), ix_accounts) in ix_trace
        .into_iter()
        .zip(ix_data_trace.into_iter())
        .zip(accounts)
    {
        // Original stack height in inner TX context.
        let inner_stack_height = ix_frame.nesting_level.saturating_add(1) as usize;

        // In sudo context, all instructions are nested under sudo (stack_height=1),
        // so we add 1 to all stack heights.
        let sudo_stack_height =
            u8::try_from(inner_stack_height.saturating_add(1)).unwrap_or(u8::MAX);

        // Remap program account index from inner to outer.
        let program_id_index = account_map
            .get(ix_frame.program_account_index_in_tx as usize)
            .copied()
            .unwrap_or(0);

        // Remap instruction account indices from inner to outer.
        let account_indices: Vec<u8> = ix_accounts
            .iter()
            .map(|acc| {
                account_map
                    .get(acc.index_in_transaction as usize)
                    .copied()
                    .unwrap_or(0)
            })
            .collect();

        result.push(InnerInstruction {
            instruction: CompiledInstruction::new_from_raw_parts(
                program_id_index,
                ix_data.into_owned(),
                account_indices,
            ),
            stack_height: sudo_stack_height,
        });
    }

    // Return as single-element list since sudo is the only top-level instruction.
    vec![result]
}

fn build_sudo_logs(
    inner_logs: Option<TransactionLogMessages>,
    inner_status: &Result<(), TransactionError>,
) -> Option<TransactionLogMessages> {
    let inner_logs = inner_logs?;

    let mut logs = Vec::with_capacity(inner_logs.len() + 2);

    // CPI Start. Sudo program invoked at stack height 1.
    logs.push(format!("Program {} invoke [1]", SUDO_PROGRAM_ID));

    // All inner logs.
    logs.extend(inner_logs);

    // CPI End. Sudo program success/failure.
    match inner_status {
        Ok(()) => logs.push(format!("Program {SUDO_PROGRAM_ID} success")),
        // TODO: Observing a failed transaction that doesn't abort the wrapper is maybe
        // gonna break some people.
        Err(e) => logs.push(format!(
            "Program {SUDO_PROGRAM_ID} failed: inner transaction error: {e:?}",
        )),
    }

    Some(logs)
}

fn build_sudo_inner_instructions(
    inner_instructions: Option<InnerInstructionsList>,
    _account_map: &[u8],
) -> Option<InnerInstructionsList> {
    let inner_instructions = inner_instructions?;

    // Filter out if all inner instruction lists are empty.
    let has_instructions = inner_instructions.iter().any(|list| !list.is_empty());

    has_instructions.then_some(inner_instructions)
}
