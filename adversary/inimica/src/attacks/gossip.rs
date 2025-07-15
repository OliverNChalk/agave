//! Attack related to the gossip

use {crate::report_attack_execution, solana_metrics::metrics::MetricsSender};

pub mod args;
pub mod flood_sigverify;

use args::GossipCli;

pub async fn run(
    metrics: &impl MetricsSender,
    json_rpc_url: &str,
    args: GossipCli,
) -> Result<(), String> {
    match args {
        GossipCli::FloodSigverify(args) => {
            report_attack_execution(
                metrics,
                "gossip_flood_sigverify",
                flood_sigverify::run(json_rpc_url, args),
            )
            .await
        }
    }
}
