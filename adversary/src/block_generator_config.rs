use {
    crate::accounts_file::{AccountsFile, AccountsFileRaw},
    log::debug,
    std::{path::PathBuf, sync::Arc},
};

#[derive(Clone, Debug)]
pub enum BlockGeneratorAccountsSource {
    /// Used for private cluster or testnet.
    AccountsPath(PathBuf),
    /// accounts provided at genesis. Used for local cluster tests.
    /// `Arc` to make it clonable which is needed for `validator::new()`.
    Genesis(Arc<AccountsFile>),
}

/// Configuration for the block generator invalidator for replay.
#[derive(Clone, Debug)]
pub struct BlockGeneratorConfig {
    pub accounts: BlockGeneratorAccountsSource,
}

impl From<BlockGeneratorAccountsSource> for Arc<AccountsFile> {
    fn from(block_generator_config: BlockGeneratorAccountsSource) -> Arc<AccountsFile> {
        match block_generator_config {
            BlockGeneratorAccountsSource::AccountsPath(file_name) => {
                let file_content = std::fs::read_to_string(file_name)
                    .expect("Failed to read the accounts file.\nPath: {file_name}");
                let accounts = serde_json::from_str::<AccountsFileRaw>(&file_content)
                    .expect(
                        "Failed to parse accounts file.\nPath: \
                         {file_name}\nContent:\n{file_content}",
                    )
                    .into();
                Arc::new(accounts)
            }
            BlockGeneratorAccountsSource::Genesis(account_file) => {
                debug!(
                    "Saving accounts for {} starting keypairs into 'payers' group",
                    account_file.payers.len()
                );
                account_file
            }
        }
    }
}
