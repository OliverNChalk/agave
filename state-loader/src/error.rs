//! Meta error which wraps all the submodule errors.
use {crate::accounts_creator::AccountsCreatorError, thiserror::Error};

#[allow(dead_code)] // fixme faycel
#[derive(Debug, Error)]
pub enum StateLoaderError {
    #[error(transparent)]
    AccountsCreatorError(#[from] AccountsCreatorError),

    #[error("Failed to read keypair file")]
    KeypairReadFailure,

    #[error("Accounts validation failed")]
    AccountsValidationFailure,

    #[error("Could not find validator identity among staked nodes")]
    FindValidatorIdentityFailure,
}
