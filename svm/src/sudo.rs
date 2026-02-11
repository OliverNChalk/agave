use {
    crate::{
        account_loader::LoadedTransaction,
        transaction_error_metrics::TransactionErrorMetrics,
        transaction_execution_result::{ExecutedTransaction, TransactionExecutionDetails},
        transaction_processor::{TransactionProcessingConfig, TransactionProcessingEnvironment},
    },
    agave_transaction_view::{
        resolved_transaction_view::ResolvedTransactionView, transaction_data::TransactionData,
        transaction_version::TransactionVersion, transaction_view::TransactionView,
    },
    solana_account::{AccountSharedData, ReadableAccount},
    solana_address_lookup_table_interface::state::AddressLookupTable,
    solana_clock::Slot,
    solana_instruction::error::InstructionError,
    solana_message::{AccountKeys, v0::LoadedAddresses},
    solana_program_runtime::{
        execution_budget::SVMTransactionExecutionCost, loaded_programs::ProgramCacheForTxBatch,
        sysvar_cache::SysvarCache,
    },
    solana_pubkey::{Pubkey, pubkey},
    solana_svm_callback::TransactionProcessingCallback,
    solana_svm_timings::ExecuteTimings,
    solana_svm_transaction::{svm_message::SVMMessage, svm_transaction::SVMTransaction},
    solana_transaction::{TransactionResult, sanitized::MAX_TX_ACCOUNT_LOCKS},
    solana_transaction_error::TransactionError,
    std::{
        collections::{HashMap, HashSet},
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
    ) {
        Some(execution_details) => todo!(),
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

    // Verify inner TX signatures.
    for (i, signature) in inner.signatures().iter().enumerate() {
        let signer_pubkey = inner.static_account_keys().get(i)?;
        if !signature.verify(signer_pubkey.as_ref(), inner.message_data()) {
            return None;
        }
    }

    // Ensure that our inner <> outer key mapping is consistent.
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

    todo!()
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
    fn deserialize(buf: &[u8]) -> Option<Self> {
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
