//! Attacks that this client may perform.

pub mod args;
pub mod program_runtime;

use args::AttackCli;

pub async fn run(json_rpc_url: &str, args: AttackCli) -> Result<(), String> {
    match args {
        AttackCli::ProgramRuntime(args) => program_runtime::run(json_rpc_url, args).await,
    }
}
