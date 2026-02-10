use {
    crate::{
        account_loader::LoadedTransaction,
        transaction_error_metrics::TransactionErrorMetrics,
        transaction_execution_result::{ExecutedTransaction, TransactionExecutionDetails},
        transaction_processor::{TransactionProcessingConfig, TransactionProcessingEnvironment},
    },
    solana_instruction::{TRANSACTION_LEVEL_STACK_HEIGHT, error::InstructionError},
    solana_program_runtime::{
        execution_budget::SVMTransactionExecutionCost, loaded_programs::ProgramCacheForTxBatch,
        sysvar_cache::SysvarCache,
    },
    solana_pubkey::{Pubkey, pubkey},
    solana_svm_callback::TransactionProcessingCallback,
    solana_svm_timings::ExecuteTimings,
    solana_svm_transaction::svm_transaction::SVMTransaction,
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
    let sudo_ix = tx.instructions_iter().next()?;
    let sudo_ix = SudoInstructionData::deserialize(sudo_ix.data)?;

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
