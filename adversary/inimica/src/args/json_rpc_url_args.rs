use {
    clap_v4::{self as clap, Args},
    solana_clap_v3_utils::input_parsers::parse_url_or_moniker,
};

/// A common argument used by multiple different commands.
#[derive(Args, Debug)]
pub struct JsonRpcUrlArgs {
    #[arg(long, value_name = "URL_OR_MONIKER", value_parser = parse_url_or_moniker)]
    /// URL for Solana's JSON RPC or moniker (or their first letter):
    ///     [mainnet-beta, testnet, devnet, localhost]
    pub json_rpc_url: String,
}
