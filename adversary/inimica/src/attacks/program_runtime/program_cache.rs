//! Attack related to the program cache.

pub mod args;
pub mod unloaded_program_invocation;

use args::ProgramCacheCli;

pub async fn run(json_rpc_url: &str, args: ProgramCacheCli) -> Result<(), String> {
    match args {
        ProgramCacheCli::UnloadedProgramInvocation(args) => {
            unloaded_program_invocation::run(json_rpc_url, args).await
        }
    }
}
