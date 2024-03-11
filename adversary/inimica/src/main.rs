use {clap::Parser, log::error};

pub mod args;
pub mod attacks;
pub(crate) mod blockhash_cache;
pub mod deploy_program;
pub(crate) mod programs;

use args::{json_rpc_url_args::JsonRpcUrlArgs, Command, InimicaCli};

#[tokio::main]
async fn main() {
    solana_logger::setup();

    let cli = InimicaCli::parse();
    let res = run(cli).await;

    if let Err(err) = res {
        error!("Failed: {err}");
    }
}

async fn run(args: InimicaCli) -> Result<(), String> {
    let InimicaCli {
        json_rpc_url: JsonRpcUrlArgs { json_rpc_url },
        command,
    } = args;

    match command {
        Command::Attack(args) => attacks::run(&json_rpc_url, args).await,
    }
}
