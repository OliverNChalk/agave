use {
    crate::accounts_file::{AccountsFile, AccountsFileRaw},
    log::debug,
    std::sync::Arc,
};

/// Defines possible ways to specify setup accounts:
/// * read accounts from file (used for private cluster or testnet)
/// * use accounts provided as part of genesis (used for local cluster tests)
#[derive(Clone, Debug)]
pub enum BlockGeneratorAccountsOption {
    AccountsPath(String),
    // Arc to make it clonable which is needed for validator::new
    Accounts(Arc<AccountsFile>),
}

/// Configuration for the block generator invalidator for replay.
#[derive(Clone, Debug)]
pub struct BlockGeneratorConfig {
    pub accounts: BlockGeneratorAccountsOption,
}

impl From<BlockGeneratorAccountsOption> for Arc<AccountsFile> {
    fn from(block_generator_config: BlockGeneratorAccountsOption) -> Arc<AccountsFile> {
        match block_generator_config {
            BlockGeneratorAccountsOption::AccountsPath(file_name) => {
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
            BlockGeneratorAccountsOption::Accounts(account_file) => {
                debug!(
                    "Saving accounts for {} starting keypairs into 'payers' group",
                    account_file.payers.len()
                );
                account_file
            }
        }
    }
}
