//! Attack related to the program runtime.

pub mod args;
pub mod program_cache;

use args::ProgramRuntimeCli;

pub async fn run(json_rpc_url: &str, args: ProgramRuntimeCli) -> Result<(), String> {
    match args {
        ProgramRuntimeCli::ProgramCache(args) => program_cache::run(json_rpc_url, args).await,
    }
}
