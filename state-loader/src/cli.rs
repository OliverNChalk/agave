use {
    crate::range::Range,
    clap::{crate_description, crate_name, crate_version, Args, Parser, Subcommand},
    solana_clap_v3_utils::{
        input_parsers::parse_url_or_moniker, input_validators::normalize_to_url_if_moniker,
    },
    solana_commitment_config::CommitmentConfig,
    solana_pubkey::Pubkey,
    solana_sdk_ids::system_program,
    solana_system_interface::MAX_PERMITTED_DATA_LENGTH,
    std::path::PathBuf,
};

fn normalize_to_url(addr: &str) -> Result<String, &'static str> {
    Ok(normalize_to_url_if_moniker(addr))
}

#[derive(Parser, Debug, PartialEq, Eq)]
#[clap(name = crate_name!(),
    version = crate_version!(),
    about = crate_description!(),
    rename_all = "kebab-case"
)]
pub struct StateLoaderCliParameters {
    #[clap(
        long = "url",
        short = 'u',
        validator = parse_url_or_moniker,
        value_parser = normalize_to_url,
        help = "URL for Solana's JSON RPC or moniker (or their first letter):\n\
        [mainnet-beta, testnet, devnet, localhost]"
    )]
    pub json_rpc_url: String,

    #[clap(
        long,
        default_value = "confirmed",
        possible_values = &["processed", "confirmed", "finalized"],
        help = "Block commitment config for getting latest blockhash.\n"
    )]
    pub commitment_config: CommitmentConfig,

    // Cannot use value_parser to read keypair file because Keypair is not Clone.
    #[clap(
        long,
        help = "Keypair file of authority. If not provided, create a new one.\nIf authority has \
                insufficient funds, client will try airdrop."
    )]
    pub authority: Option<PathBuf>,

    #[clap(
        long,
        help = "Validate the created accounts number, size, balance.\nMight be time consuming, so \
                recommended only for debugging purposes."
    )]
    pub validate_accounts: bool,

    #[clap(subcommand)]
    pub command: Command,
}

#[derive(Args, Debug, PartialEq, Eq, Clone)]
#[clap(rename_all = "kebab-case")]
pub struct WriteAccounts {
    #[clap(long, help = "File to save the created accounts into")]
    pub accounts_file: PathBuf,

    #[clap(flatten)]
    pub account_params: AccountParams,
}

#[derive(Args, Debug, PartialEq, Eq, Clone)]
#[clap(rename_all = "kebab-case")]
pub struct ReadAccounts {
    #[clap(long, help = "File to read the accounts from")]
    pub accounts_file: PathBuf,
}

#[derive(Args, Copy, Clone, Debug, PartialEq, Eq)]
#[clap(rename_all = "kebab-case")]
pub struct AccountParams {
    #[clap(
        long,
        default_value = "2048",
        help = "Number of sized accounts to create."
    )]
    pub num_accounts: usize,

    #[clap(long, default_value = "1024", help = "Number of payer accounts.")]
    pub num_payers: usize,

    #[clap(
        long,
        default_value = "1",
        validator = validate_account_size,
        help = "Account size (bytes) in the format '<value>|[<value>,<value>]'.\n\
                If interval is specified, the uniform distribution will be used.\n"
    )]
    pub account_size: Range,

    #[clap(
        long,
        default_value = "1",
        help = "Payer account balance in SOL,\nused to fund creation of other accounts and for \
                transactions.\n"
    )]
    pub payer_account_balance: u64,

    #[clap(
        long,
        default_value_t = system_program::id(),
        help = "Program that owns sized accounts, by default system program.\n"
    )]
    pub account_owner: Pubkey,
}

#[derive(Subcommand, Debug, PartialEq, Eq)]
pub enum Command {
    #[clap(about = "Create accounts and save them to a file, skipping the execution")]
    WriteAccounts(WriteAccounts),
    #[clap(about = "Read accounts from a file")]
    ReadAccounts(ReadAccounts),
}

pub fn build_cli_parameters() -> StateLoaderCliParameters {
    StateLoaderCliParameters::parse()
}

fn validate_account_size(range: &str) -> Result<(), String> {
    let range: Range = range.parse()?;
    if range.max > MAX_PERMITTED_DATA_LENGTH as usize {
        Err("Account size cannot be greater than 10MB".to_string())
    } else {
        Ok(())
    }
}

#[cfg(feature = "dev-context-only-utils")]
pub fn get_common_account_params() -> (Vec<&'static str>, AccountParams) {
    (
        vec![
            "--num-payers",
            "256",
            "--num-accounts",
            "1024",
            "--account-size",
            "[128,512]",
            "--payer-account-balance",
            "1",
        ],
        AccountParams {
            num_accounts: 1024,
            num_payers: 256,
            account_size: Range { min: 128, max: 512 },
            payer_account_balance: 1,
            account_owner: system_program::id(),
        },
    )
}

#[cfg(test)]
mod tests {
    use {super::*, clap::Parser};

    #[test]
    fn test_write_accounts_command() {
        let keypair_file_name = "/home/testUser/masterKey.json";
        let accounts_file_name = "/home/testUser/accountsFile.json";

        let mut args = vec![
            "test",
            "-ul",
            "--authority",
            keypair_file_name,
            "write-accounts",
            "--accounts-file",
            accounts_file_name,
        ];

        let (account_args, account_params) = get_common_account_params();
        args.extend(account_args.iter());

        let expected_parameters = StateLoaderCliParameters {
            json_rpc_url: "http://localhost:8899".to_string(),
            commitment_config: CommitmentConfig::confirmed(),
            command: Command::WriteAccounts(WriteAccounts {
                accounts_file: accounts_file_name.into(),
                account_params,
            }),
            authority: Some(PathBuf::from(&keypair_file_name)),
            validate_accounts: false,
        };
        let cli = StateLoaderCliParameters::try_parse_from(args);
        assert!(cli.is_ok(), "Unexpected error {:?}", cli.err());
        let actual_parameters = cli.unwrap();

        assert_eq!(actual_parameters, expected_parameters);
    }

    #[test]
    fn test_read_accounts_command() {
        let keypair_file_name = "/home/testUser/masterKey.json";
        let accounts_file_name = "/home/testUser/accountsFile.json";

        let args = vec![
            "test",
            "-ul",
            "--authority",
            keypair_file_name,
            "read-accounts",
            "--accounts-file",
            accounts_file_name,
        ];

        let expected_parameters = StateLoaderCliParameters {
            json_rpc_url: "http://localhost:8899".to_string(),
            commitment_config: CommitmentConfig::confirmed(),
            command: Command::ReadAccounts(ReadAccounts {
                accounts_file: accounts_file_name.into(),
            }),
            authority: Some(PathBuf::from(&keypair_file_name)),
            validate_accounts: false,
        };
        let cli = StateLoaderCliParameters::try_parse_from(args);
        assert!(cli.is_ok(), "Unexpected error {:?}", cli.err());
        let actual_parameters = cli.unwrap();

        assert_eq!(actual_parameters, expected_parameters);
    }
}
