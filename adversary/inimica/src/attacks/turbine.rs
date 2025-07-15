//! Attack related to the turbine

use {crate::report_attack_execution, solana_metrics::metrics::MetricsSender};

pub mod args;
pub mod flood_sigverify;

use args::TurbineCli;

pub async fn run(
    metrics: &impl MetricsSender,
    json_rpc_url: &str,
    args: TurbineCli,
) -> Result<(), String> {
    match args {
        TurbineCli::FloodSigverify(args) => {
            report_attack_execution(
                metrics,
                "turbine_flood_sigverify",
                flood_sigverify::run(json_rpc_url, args),
            )
            .await
        }
    }
}
