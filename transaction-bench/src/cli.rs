use {
    crate::range::Range,
    clap::{crate_description, crate_name, crate_version, Args, Parser},
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
        help = "URL for Solana's JSON RPC or moniker (or their first letter): \
        [mainnet-beta, testnet, devnet, localhost]"
    )]
    pub json_rpc_url: String,

    #[clap(
        long,
        parse(try_from_str = parse_duration),
        help = "Seconds to run benchmark, then exit; default is forever"
    )]
    pub duration: Option<Duration>,

    #[clap(
        long,
        default_value = "confirmed",
        possible_values = &["processed", "confirmed", "finalized"],
        help = "Block commitment config for getting latest blockhash"
    )]
    pub commitment_config: CommitmentConfig,

    #[clap(long, help = "Pinned address to send transactions.")]
    pub pinned_address: Option<SocketAddr>,

    // Cannot use value_parser to read keypair file because Keypair is not Clone.
    #[clap(long, help = "validator identity for staked connection")]
    pub staked_identity_file: Option<PathBuf>,

    /// Address to bind on, default will listen on all available interfaces, 0 that
    /// OS will choose the port.
    #[clap(long, help = "bind", default_value = "0.0.0.0:0")]
    pub bind: SocketAddr,

    #[clap(flatten)]
    pub transaction_params: TransactionParams,

    #[clap(flatten)]
    pub account_params: AccountParams,

    // Cannot use value_parser to read keypair file because Keypair is not Clone.
    #[clap(
        long,
        help = "Keypair file of authority. If not provided, try create one and airdrop."
    )]
    pub authority: Option<PathBuf>,

    #[clap(
        long,
        help = "Validate the created accounts number, size, balance.
        Might be time consuming, so recommended only for debugging purposes."
    )]
    pub validate_accounts: bool,

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
        help = "Number of sized accounts in transaction in format '<value>|[<value>,<value>]'.\
        If interval is specified, the uniform distribution will be used."
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
        help = "Account size (bytes) in the format '<value>|[<value>,<value>]'.\
        If interval is specified, the uniform distribution will be used."
    )]
    pub account_size: Range,

    #[clap(
        long,
        default_value = "1",
        help = "Payer account balance in SOL, used to fund creation of other accounts and for \
                transactions."
    )]
    pub payer_account_balance: u64,

    #[clap(
        long,
        default_value_t = system_program::id(),
        help = "Program that owns sized accounts, by default system program."
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

    #[test]
    fn test_typical_local_setup() {
        let keypair_file_name = "/tmp/randomfile.json";

        let cli = ClientCliParameters::try_parse_from([
            "test",
            "-ul",
            "--authority",
            keypair_file_name,
            "--staked-identity-file",
            keypair_file_name,
            "--duration",
            "120",
            "--num-payers",
            "256",
            "--num-accounts",
            "1024",
            "--account-size",
            "[128,512]",
            "--payer-account-balance",
            "1",
            "--pinned-address",
            "127.0.0.1:8009",
            "--num-accounts-per-tx",
            "[1,10]",
            "--transaction-cu-budget",
            "2000",
        ]);
        assert!(cli.is_ok());
        let actual = cli.unwrap();

        assert_eq!(
            actual,
            ClientCliParameters {
                json_rpc_url: "http://localhost:8899".to_string(),
                duration: Some(Duration::from_secs(120)),
                commitment_config: CommitmentConfig::confirmed(),
                pinned_address: Some(SocketAddr::new(
                    IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
                    8009
                )),
                staked_identity_file: Some(PathBuf::from(&keypair_file_name)),
                bind: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), 0),
                transaction_params: TransactionParams {
                    num_accounts_per_tx: Range { min: 1, max: 10 },
                    transaction_cu_budget: 2000
                },
                account_params: AccountParams {
                    num_accounts: 1024,
                    num_payers: 256,
                    account_size: Range { min: 128, max: 512 },
                    payer_account_balance: 1,
                    account_owner: system_program::id()
                },
                authority: Some(PathBuf::from(&keypair_file_name)),
                validate_accounts: false,
                num_max_open_connections: 16,
            }
        );
    }
}
