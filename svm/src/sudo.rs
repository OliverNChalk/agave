use {
    crate::{
        account_loader::LoadedTransaction,
        transaction_error_metrics::TransactionErrorMetrics,
        transaction_execution_result::ExecutedTransaction,
        transaction_processor::{TransactionProcessingConfig, TransactionProcessingEnvironment},
    },
    solana_program_runtime::{
        execution_budget::SVMTransactionExecutionCost, loaded_programs::ProgramCacheForTxBatch,
        sysvar_cache::SysvarCache,
    },
    solana_pubkey::{Pubkey, pubkey},
    solana_svm_callback::TransactionProcessingCallback,
    solana_svm_timings::ExecuteTimings,
    solana_svm_transaction::svm_transaction::SVMTransaction,
    std::sync::RwLock,
};

pub const SUDO_PROGRAM_ID: Pubkey = pubkey!("Sudo111111111111111111111111111111111111111");

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
    todo!()
}
