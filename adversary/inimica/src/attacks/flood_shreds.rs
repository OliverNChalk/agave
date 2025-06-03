//! Attacks shred flooding

use {crate::report_attack_execution, solana_metrics::metrics::MetricsSender};

pub mod args;
pub mod invalid_shreds;

use args::FloodShredsCli;

pub async fn run(
    metrics: &impl MetricsSender,
    json_rpc_url: &str,
    args: FloodShredsCli,
) -> Result<(), String> {
    match args {
        FloodShredsCli::InvalidShreds(args) => {
            report_attack_execution(
                metrics,
                "invalid-shred",
                invalid_shreds::run(json_rpc_url, args),
            )
            .await
        }
    }
}
