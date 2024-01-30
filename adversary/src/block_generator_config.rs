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

// TODO Will be replaced by AttackAccountsManager structure, see issue#215
impl BlockGeneratorConfig {
    pub fn get_accounts(&self) -> Arc<AccountsFile> {
        Arc::<AccountsFile>::from(self.accounts.clone())
    }
}

impl From<BlockGeneratorAccountsSource> for Arc<AccountsFile> {
    fn from(block_generator_config: BlockGeneratorAccountsSource) -> Arc<AccountsFile> {
        match block_generator_config {
            BlockGeneratorAccountsSource::AccountsPath(file_name) => {
                let file_content = std::fs::read_to_string(&file_name).unwrap_or_else(|err| {
                    panic!("Failed to read the accounts file.\nPath: {file_name:?}\nError: {err}")
                });
                let accounts = serde_json::from_str::<AccountsFileRaw>(&file_content)
                    .unwrap_or_else(|err| {
                        panic!(
                            "Failed to parse accounts file.\nPath: {file_name:?}\nError: \
                             {err}\nContent:\n{file_content}"
                        )
                    })
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
