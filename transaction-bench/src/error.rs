//! Meta error which wraps all the submodule errors.
use {
    solana_tpu_client_next::{leader_updater::LeaderUpdaterError, ConnectionWorkersSchedulerError},
    thiserror::Error,
};

#[derive(Debug, Error)]
pub enum BenchClientError {
    #[error(transparent)]
    AccountsCreatorError(#[from] crate::accounts_creator::AccountsCreatorError),

    #[error(transparent)]
    ConnectionTasksSchedulerError(#[from] ConnectionWorkersSchedulerError),

    #[error("Failed to read keypair file")]
    KeypairReadFailure,

    #[error("Accounts validation failed")]
    AccountsValidationFailure,

    #[error("Could not find validator identity among staked nodes")]
    FindValidatorIdentityFailure,

    #[error("Leader updater failed")]
    LeaderUpdaterError(#[from] LeaderUpdaterError),
}
