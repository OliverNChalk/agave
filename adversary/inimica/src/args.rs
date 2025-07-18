use {
    crate::attacks::args::AttackCli,
    clap_v4::{self as clap, Parser, Subcommand},
};

pub mod accounts_file;
pub mod json_rpc_url_args;

pub use accounts_file::AccountsFileArgs;
use json_rpc_url_args::JsonRpcUrlArgs;

/// Holds global configuration, plus an attack selection.
#[derive(Parser, Debug)]
#[command(version, about)]
pub struct InimicaCli {
    #[command(flatten)]
    pub json_rpc_url: JsonRpcUrlArgs,

    #[command(subcommand)]
    pub command: Command,
}

/// A specific action to perform.
#[derive(Subcommand, Debug)]
pub enum Command {
    #[command(subcommand)]
    /// Run an attack against the cluster.
    Attack(AttackCli),
}
