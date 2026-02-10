use {
    crate::{
        account_loader::LoadedTransaction,
        transaction_error_metrics::TransactionErrorMetrics,
        transaction_execution_result::{ExecutedTransaction, TransactionExecutionDetails},
        transaction_processor::{TransactionProcessingConfig, TransactionProcessingEnvironment},
    },
    agave_transaction_view::{
        resolved_transaction_view::ResolvedTransactionView,
        transaction_data::TransactionData,
        transaction_version::TransactionVersion,
        transaction_view::{SanitizedTransactionView, TransactionView},
    },
    solana_clock::Slot,
    solana_instruction::{TRANSACTION_LEVEL_STACK_HEIGHT, error::InstructionError},
    solana_message::v0::LoadedAddresses,
    solana_program_runtime::{
        execution_budget::SVMTransactionExecutionCost, loaded_programs::ProgramCacheForTxBatch,
        sysvar_cache::SysvarCache,
    },
    solana_pubkey::{Pubkey, pubkey},
    solana_svm_callback::TransactionProcessingCallback,
    solana_svm_timings::ExecuteTimings,
    solana_svm_transaction::svm_transaction::SVMTransaction,
    solana_transaction::{
        sanitized::MessageHash, versioned::sanitized::SanitizedVersionedTransaction,
    },
    solana_transaction_error::TransactionError,
    std::{collections::HashMap, sync::RwLock},
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
    let inner_view =
        TransactionView::try_new_sanitized(sudo_ix.inner_tx_bytes.as_slice(), true, true).ok()?;

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

// Yoinked and amended from `receive_and_buffer.rs`
fn translate_to_runtime_view<D: TransactionData>(
    data: D,
    bank: &Bank,
    enable_static_instruction_limit: bool,
    transaction_account_lock_limit: usize,
    enable_instruction_accounts_limit: bool,
) -> Option<(RuntimeTransaction<ResolvedTransactionView<D>>, u64)> {
    // Parsing and basic sanitization checks
    let view = SanitizedTransactionView::try_new_sanitized(
        data,
        enable_static_instruction_limit,
        enable_instruction_accounts_limit,
    )
    .ok()?;

    let view = RuntimeTransaction::<SanitizedTransactionView<_>>::try_new(
        view,
        MessageHash::Compute,
        None,
    )
    .ok()?;

    if usize::from(view.total_num_accounts()) > transaction_account_lock_limit {
        return None;
    }

    let (loaded_addresses, deactivation_slot) = load_addresses_for_view(&view, bank)?;

    let Ok(view) = RuntimeTransaction::<ResolvedTransactionView<_>>::try_new(
        view,
        loaded_addresses,
        bank.get_reserved_account_keys(),
    ) else {
        return Err(PacketHandlingError::Sanitization);
    };

    // Validate no duplicate accounts (must be after resolution to catch ALT duplicates)
    validate_account_locks(view.account_keys(), transaction_account_lock_limit).ok()?;

    Some((view, deactivation_slot))
}

// Yoinked and amended from `receive_and_buffer.rs`
fn load_addresses_for_view<D: TransactionData>(
    view: &SanitizedTransactionView<D>,
    // TODO: Replace with AddressLoader trait on the SudoInstruction.
    bank: &Bank,
) -> Option<(Option<LoadedAddresses>, Slot)> {
    match view.version() {
        TransactionVersion::Legacy => Some((None, u64::MAX)),
        TransactionVersion::V0 => bank
            .load_addresses_from_ref(view.address_table_lookup_iter())
            .map(|(loaded_addresses, deactivation_slot)| {
                (Some(loaded_addresses), deactivation_slot)
            })
            .ok(),
    }
}

fn validate_account_locks(
    account_keys: AccountKeys,
    tx_account_lock_limit: usize,
) -> TransactionResult<()> {
    if account_keys.len() > tx_account_lock_limit {
        Err(TransactionError::TooManyAccountLocks)
    } else if has_duplicates(account_keys) {
        Err(TransactionError::AccountLoadedTwice)
    } else {
        Ok(())
    }
}
