use {
    clap_v4::Parser,
    core::future::Future,
    log::{error, Level},
    solana_metrics::{
        create_datapoint,
        metrics::{MetricsAgent, MetricsSender},
    },
};

pub mod args;
pub mod attacks;
pub(crate) mod blockhash_cache;
pub mod deploy_program;
pub(crate) mod programs;

use args::{json_rpc_url_args::JsonRpcUrlArgs, Command, InimicaCli};

#[tokio::main]
async fn main() {
    solana_logger::setup();

    let cli = InimicaCli::parse();

    let metrics = config_metrics();
    let res = run(&metrics, cli).await;
    metrics.flush();

    if let Err(err) = res {
        error!("Failed: {err}");
    }
}

async fn run(metrics: &MetricsAgent, args: InimicaCli) -> Result<(), String> {
    let InimicaCli {
        json_rpc_url: JsonRpcUrlArgs { json_rpc_url },
        command,
    } = args;

    match command {
        Command::Attack(args) => attacks::run(metrics, &json_rpc_url, args).await,
    }
}

fn config_metrics() -> MetricsAgent {
    // TODO Should we distinguish metrics sent by inimica and/or other tools with some tag we always
    // add to those metrics?
    //
    // Alternatively, we may want to use a different `host` value.
    //
    // We may also want to add CLI parameters that could overwrite or augment metrics configuration
    // taken from the `SOLANA_METRICS_CONFIG` environment variable.
    MetricsAgent::default()
}

async fn report_attack_execution<Res>(
    metrics: &impl MetricsSender,
    attack_name: &str,
    attack: impl Future<Output = Res>,
) -> Res {
    // Reporting two values next to each other creates a better signal in the UI.
    // Chronograf shows a quick change from 0 to 1, rather than a linear slope from whenever the
    // attack was disabled last time.
    metrics.submit(
        create_datapoint!(@point "adversary",
            "client" => "inimica",
            "attack" => attack_name,
            ("active", 0, i64),
        ),
        Level::Info,
    );
    metrics.submit(
        create_datapoint!(@point "adversary",
            "client" => "inimica",
            "attack" => attack_name,
            ("active", 1, i64),
        ),
        Level::Info,
    );

    let res = attack.await;

    // Similarly to the above, we report two values to create a quick change from 1 to 0.
    metrics.submit(
        create_datapoint!(@point "adversary",
            "client" => "inimica",
            "attack" => attack_name,
            ("active", 1, i64),
        ),
        Level::Info,
    );
    metrics.submit(
        create_datapoint!(@point "adversary",
            "client" => "inimica",
            "attack" => attack_name,
            ("active", 0, i64),
        ),
        Level::Info,
    );

    res
}
