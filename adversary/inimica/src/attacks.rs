//! Attacks that this client may perform.

use solana_metrics::metrics::MetricsSender;

pub mod args;
pub mod flood_shreds;
pub mod program_runtime;

use args::AttackCli;

pub async fn run(
    metrics: &impl MetricsSender,
    json_rpc_url: &str,
    args: AttackCli,
) -> Result<(), String> {
    match args {
        AttackCli::ProgramRuntime(args) => program_runtime::run(metrics, json_rpc_url, args).await,
        AttackCli::FloodShreds(args) => flood_shreds::run(metrics, json_rpc_url, args).await,
    }
}
