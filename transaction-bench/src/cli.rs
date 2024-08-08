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
    std::{net::SocketAddr, path::PathBuf},
    tokio::time::Duration,
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
pub struct ClientCliParameters {
    #[clap(
        long = "url",
        short = 'u',
        validator = parse_url_or_moniker,
        parse(try_from_str = normalize_to_url),
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

#[derive(Subcommand, Debug, PartialEq, Eq)]
pub enum Command {
    #[clap(about = "Create accounts without saving them and run")]
    Run {
        #[clap(flatten)]
        account_params: AccountParams,

        #[clap(flatten)]
        execution_params: ExecutionParams,

        #[clap(flatten)]
        transaction_params: TransactionParams,
    },

    #[clap(about = "Read accounts from provided accounts file and run")]
    ReadAccountsRun {
        #[clap(long, help = "File with saved accounts")]
        accounts_file: PathBuf,

        #[clap(flatten)]
        execution_params: ExecutionParams,

        #[clap(flatten)]
        transaction_params: TransactionParams,
    },

    #[clap(about = "Create accounts and save them to a file, skipping the execution")]
    WriteAccounts {
        #[clap(long, help = "File to save the created accounts into")]
        accounts_file: PathBuf,

        #[clap(flatten)]
        account_params: AccountParams,
    },
}

#[derive(Args, Clone, Debug, PartialEq, Eq)]
#[clap(rename_all = "kebab-case")]
pub struct ExecutionParams {
    // Cannot use value_parser to read keypair file because Keypair is not Clone.
    #[clap(long, help = "validator identity for staked connection.")]
    pub staked_identity_file: Option<PathBuf>,

    /// Address to bind on, default will listen on all available interfaces, 0 that
    /// OS will choose the port.
    #[clap(long, help = "bind", default_value = "0.0.0.0:0")]
    pub bind: SocketAddr,

    #[clap(
        long,
        parse(try_from_str = parse_duration),
        help = "If specified, limits the benchmark execution to the specified duration."
    )]
    pub duration: Option<Duration>,

    #[clap(long, help = "Pinned address to send transactions.")]
    pub pinned_address: Option<SocketAddr>,

    #[clap(
        long,
        default_value_t = 16,
        help = "Max number of connections to keep open."
    )]
    pub num_max_open_connections: usize,
}

#[derive(Args, Clone, Debug, PartialEq, Eq)]
#[clap(rename_all = "kebab-case")]
pub struct TransactionParams {
    #[clap(
        long,
        default_value = "1",
        validator = validate_num_accounts_per_tx,
        help = "Number of sized accounts in transaction in format '<value>|[<value>,<value>]'.\n\
                If interval is specified, the uniform distribution will be used.\n"
    )]
    pub num_accounts_per_tx: Range,

    #[clap(long, default_value = "20000", help = "Transaction CU budget.")]
    pub transaction_cu_budget: u32,
}

#[derive(Args, Copy, Clone, Debug, PartialEq, Eq)]
#[clap(rename_all = "kebab-case")]
pub struct AccountParams {
    #[clap(
        long,
        default_value = "1024",
        help = "Number of sized accounts to create."
    )]
    pub num_accounts: usize,

    #[clap(long, default_value = "256", help = "Number of payer accounts.")]
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

fn parse_duration(s: &str) -> Result<Duration, &'static str> {
    s.parse::<u64>()
        .map(Duration::from_secs)
        .map_err(|_| "failed to parse duration")
}

fn validate_account_size(range: &str) -> Result<(), String> {
    let range: Range = range.parse()?;
    if range.max > MAX_PERMITTED_DATA_LENGTH as usize {
        Err("Account size cannot be greater than 10MB".to_string())
    } else {
        Ok(())
    }
}

fn validate_num_accounts_per_tx(range: &str) -> Result<(), String> {
    let range: Range = range.parse()?;
    if range.max > 62 {
        Err("One transaction cannot have more than 62 accounts without using ALT.".to_string())
    } else {
        Ok(())
    }
}

pub fn build_cli_parameters() -> ClientCliParameters {
    ClientCliParameters::parse()
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        clap::Parser,
        std::net::{IpAddr, Ipv4Addr},
    };

    fn get_common_execution_params(keypair_file_name: &str) -> (Vec<&str>, ExecutionParams) {
        (
            vec![
                "--staked-identity-file",
                keypair_file_name,
                "--duration",
                "120",
                "--pinned-address",
                "127.0.0.1:8009",
            ],
            ExecutionParams {
                staked_identity_file: Some(PathBuf::from(&keypair_file_name)),
                bind: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), 0),
                duration: Some(Duration::from_secs(120)),
                pinned_address: Some(SocketAddr::new(
                    IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
                    8009,
                )),
                num_max_open_connections: 16,
            },
        )
    }

    fn get_common_account_params() -> (Vec<&'static str>, AccountParams) {
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

    #[test]
    fn test_run_command() {
        let keypair_file_name = "/home/testUser/masterKey.json";

        let mut args = vec![
            "test",
            "-ul",
            "--authority",
            keypair_file_name,
            "run",
            "--num-accounts-per-tx",
            "[1,10]",
            "--transaction-cu-budget",
            "2000",
        ];
        let (exec_args, execution_params) = get_common_execution_params(keypair_file_name);
        args.extend(exec_args.iter());
        let (account_args, account_params) = get_common_account_params();
        args.extend(account_args.iter());

        let expected_parameters = ClientCliParameters {
            json_rpc_url: "http://localhost:8899".to_string(),
            commitment_config: CommitmentConfig::confirmed(),
            command: Command::Run {
                transaction_params: TransactionParams {
                    num_accounts_per_tx: Range { min: 1, max: 10 },
                    transaction_cu_budget: 2000,
                },
                account_params,
                execution_params,
            },
            authority: Some(PathBuf::from(&keypair_file_name)),
            validate_accounts: false,
        };
        let actual = ClientCliParameters::try_parse_from(args).unwrap();

        assert_eq!(actual, expected_parameters);
    }

    #[test]
    fn test_read_accounts_run_command() {
        let keypair_file_name = "/home/testUser/masterKey.json";
        let accounts_file_name = "/home/testUser/accountsFile.json";

        let mut args = vec![
            "test",
            "-ul",
            "--authority",
            keypair_file_name,
            "read-accounts-run",
            "--accounts-file",
            accounts_file_name,
            "--num-accounts-per-tx",
            "[1,10]",
            "--transaction-cu-budget",
            "2000",
        ];
        let (exec_args, execution_params) = get_common_execution_params(keypair_file_name);
        args.extend(exec_args.iter());

        let expected_parameters = ClientCliParameters {
            json_rpc_url: "http://localhost:8899".to_string(),
            commitment_config: CommitmentConfig::confirmed(),
            command: Command::ReadAccountsRun {
                accounts_file: accounts_file_name.into(),
                transaction_params: TransactionParams {
                    num_accounts_per_tx: Range { min: 1, max: 10 },
                    transaction_cu_budget: 2000,
                },
                execution_params,
            },
            authority: Some(PathBuf::from(&keypair_file_name)),
            validate_accounts: false,
        };
        let cli = ClientCliParameters::try_parse_from(args);
        assert!(cli.is_ok(), "Unexpected error {:?}", cli.err());
        let actual = cli.unwrap();

        assert_eq!(actual, expected_parameters);
    }

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

        let expected_parameters = ClientCliParameters {
            json_rpc_url: "http://localhost:8899".to_string(),
            commitment_config: CommitmentConfig::confirmed(),
            command: Command::WriteAccounts {
                accounts_file: accounts_file_name.into(),
                account_params,
            },
            authority: Some(PathBuf::from(&keypair_file_name)),
            validate_accounts: false,
        };
        let cli = ClientCliParameters::try_parse_from(args);
        assert!(cli.is_ok(), "Unexpected error {:?}", cli.err());
        let actual = cli.unwrap();

        assert_eq!(actual, expected_parameters);
    }

    /// Check that cannot use `write` subcommand together with parameters from `TransactionParams`
    #[test]
    fn test_write_accounts_file_conflict() {
        let keypair_file_name = "/home/testUser/masterKey.json";
        let accounts_file_name = "/home/testUser/accountsFile.json";

        let mut args = vec![
            "test",
            "-ul",
            "--authority",
            keypair_file_name,
            "write-accounts",
            "--num-accounts-per-tx",
            "100",
            "--accounts-file",
            accounts_file_name,
        ];

        let (account_args, _account_params) = get_common_account_params();
        args.extend(account_args.iter());

        let cli = ClientCliParameters::try_parse_from(args);
        assert!(cli.is_err());
    }
}
