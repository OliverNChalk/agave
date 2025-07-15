//! Attacks that this client may perform.

use solana_metrics::metrics::MetricsSender;

pub mod args;
pub mod gossip;
pub mod program_runtime;
pub mod turbine;

use args::AttackCli;

pub async fn run(
    metrics: &impl MetricsSender,
    json_rpc_url: &str,
    args: AttackCli,
) -> Result<(), String> {
    match args {
        AttackCli::Gossip(args) => gossip::run(metrics, json_rpc_url, args).await,
        AttackCli::ProgramRuntime(args) => program_runtime::run(metrics, json_rpc_url, args).await,
        AttackCli::Turbine(args) => turbine::run(metrics, json_rpc_url, args).await,
    }
}
