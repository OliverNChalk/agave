//! Attack related to the program cache.

use solana_metrics::metrics::MetricsSender;

pub mod args;
pub mod unloaded_program_invocation;

use args::ProgramCacheCli;

pub async fn run(
    metrics: &impl MetricsSender,
    json_rpc_url: &str,
    args: ProgramCacheCli,
) -> Result<(), String> {
    match args {
        ProgramCacheCli::UnloadedProgramInvocation(args) => {
            unloaded_program_invocation::run(metrics, json_rpc_url, args).await
        }
    }
}
