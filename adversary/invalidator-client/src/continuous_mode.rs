use {
    crate::adversary,
    log::*,
    std::{thread, time::Duration},
};

struct AdversaryScenario<'a> {
    start_fn: Box<dyn Fn() -> Result<(), String> + 'a>,
    stop_fn: Box<dyn Fn() -> Result<(), String> + 'a>,
}

pub fn run_continuous_mode(
    rpc_endpoint_url: &str,
    scenario_run_duration: Duration,
    rest_between_scenarios_duration: Duration,
) -> Result<(), String> {
    let rpc_endpoint_url = rpc_endpoint_url.to_owned();
    let mut adversary_scenarios: Vec<(&str, AdversaryScenario)> = Vec::new();

    adversary_scenarios.push((
        "send_duplicate_blocks",
        AdversaryScenario {
            start_fn: Box::new(|| {
                adversary::leader_block::configure_send_duplicate_blocks_enable(
                    rpc_endpoint_url.clone(),
                )
            }),
            stop_fn: Box::new(|| {
                adversary::leader_block::configure_send_duplicate_blocks_disable(
                    rpc_endpoint_url.clone(),
                )
            }),
        },
    ));

    adversary_scenarios.push((
        "invalidate_leader_block",
        AdversaryScenario {
            start_fn: Box::new(|| {
                adversary::leader_block::configure_invalidate_leader_block_enable(
                    rpc_endpoint_url.clone(),
                )
            }),
            stop_fn: Box::new(|| {
                adversary::leader_block::configure_invalidate_leader_block_disable(
                    rpc_endpoint_url.clone(),
                )
            }),
        },
    ));

    adversary_scenarios.push((
        "drop_turbine_votes",
        AdversaryScenario {
            start_fn: Box::new(|| {
                adversary::drop_turbine_votes::configure_drop_turbine_votes_enable(
                    rpc_endpoint_url.clone(),
                )
            }),
            stop_fn: Box::new(|| {
                adversary::drop_turbine_votes::configure_drop_turbine_votes_disable(
                    rpc_endpoint_url.clone(),
                )
            }),
        },
    ));

    adversary_scenarios.push((
        "repair_packet_flood",
        AdversaryScenario {
            start_fn: Box::new(|| {
                adversary::repair::configure_repair_packet_flood_enable(rpc_endpoint_url.clone())
            }),
            stop_fn: Box::new(|| {
                adversary::repair::configure_repair_packet_flood_disable(rpc_endpoint_url.clone())
            }),
        },
    ));

    loop {
        for (label, adversary_scenario) in &adversary_scenarios {
            if let Err(e) = (adversary_scenario.start_fn)() {
                error!("Failed to start scenario {label}: {e}");
                continue;
            }
            info!("Running scenario {label} for {scenario_run_duration:?}...");
            thread::sleep(scenario_run_duration);
            if let Err(e) = (adversary_scenario.stop_fn)() {
                error!("Failed to stop scenario {label}: {e}");
            }
            info!("Completed scenario {label}. Resting for {rest_between_scenarios_duration:?}...");
            thread::sleep(rest_between_scenarios_duration);
        }
    }

    #[allow(unreachable_code)]
    Ok(())
}
