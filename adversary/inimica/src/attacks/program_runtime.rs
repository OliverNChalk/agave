//! Attack related to the program runtime.

use solana_metrics::metrics::MetricsSender;

pub mod args;
pub mod program_cache;

use args::ProgramRuntimeCli;

pub async fn run(
    metrics: &impl MetricsSender,
    json_rpc_url: &str,
    args: ProgramRuntimeCli,
) -> Result<(), String> {
    match args {
        ProgramRuntimeCli::ProgramCache(args) => {
            program_cache::run(metrics, json_rpc_url, args).await
        }
    }
}
