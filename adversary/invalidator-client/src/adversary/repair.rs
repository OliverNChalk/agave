use {
    super::Command,
    clap::{value_t_or_exit, ArgMatches},
    solana_adversary::adversary_feature_set::{
        repair_packet_flood::{
            AdversarialConfig as RepairPacketFloodConfig, FloodConfig, FloodStrategy,
            PeerIdentifier,
        },
        repair_parameters::AdversarialConfig as RepairParametersConfig,
    },
};

pub const DEFAULT_PACKETS_PER_PEER_PER_ITERATION: &str = "10";
pub const DEFAULT_ITERATION_DELAY_US: &str = "1000000";

impl Command for RepairPacketFloodConfig {
    const RPC_METHOD: &'static str = "configureRepairPacketFlood";
}

impl Command for RepairParametersConfig {
    const RPC_METHOD: &'static str = "configureRepairParameters";
}

pub fn configure_repair_packet_flood_enable(rpc_endpoint_url: String) -> Result<(), String> {
    configure_repair_packet_flood(
        &rpc_endpoint_url,
        RepairPacketFloodConfig {
            configs: vec![FloodConfig {
                flood_strategy: FloodStrategy::MinimalPackets,
                packets_per_peer_per_iteration: DEFAULT_PACKETS_PER_PEER_PER_ITERATION
                    .parse()
                    .unwrap(),
                iteration_delay_us: DEFAULT_ITERATION_DELAY_US.parse().unwrap(),
                target: None,
            }],
        },
    )
}

pub fn configure_repair_packet_flood_disable(rpc_endpoint_url: String) -> Result<(), String> {
    configure_repair_packet_flood(
        &rpc_endpoint_url,
        RepairPacketFloodConfig { configs: vec![] },
    )
}

pub fn configure_repair_packet_flood_args(
    rpc_endpoint_url: &str,
    sub_matches: &ArgMatches<'_>,
) -> Result<(), String> {
    let disable = value_t_or_exit!(sub_matches, "disable", bool);
    let flood_strategy: FloodStrategy = serde_json::from_str(&format!(
        r#""{}""#,
        value_t_or_exit!(sub_matches, "flood_strategy", String)
    ))
    .map_err(|e| format!("Error converting to enum from string: {e}"))?;
    let packets_per_peer_per_iteration =
        value_t_or_exit!(sub_matches, "packets_per_peer_per_iteration", u64);
    let iteration_delay_us = value_t_or_exit!(sub_matches, "iteration_delay_us", u64);
    let target = sub_matches
        .value_of("target")
        .map(|pubkey| PeerIdentifier::Pubkey(pubkey.to_owned()));

    let configs = if disable {
        vec![]
    } else {
        vec![FloodConfig {
            flood_strategy,
            packets_per_peer_per_iteration: (packets_per_peer_per_iteration as u32),
            iteration_delay_us,
            target,
        }]
    };
    configure_repair_packet_flood(rpc_endpoint_url, RepairPacketFloodConfig { configs })
}

pub fn configure_repair_packet_flood(
    rpc_endpoint_url: &str,
    repair_packet_flood_config: RepairPacketFloodConfig,
) -> Result<(), String> {
    repair_packet_flood_config.send(rpc_endpoint_url)
}

pub fn configure_repair_parameters_args(
    rpc_endpoint_url: &str,
    sub_matches: &ArgMatches<'_>,
) -> Result<(), String> {
    let serve_repair_max_requests_per_iteration = sub_matches
        .value_of("serve_repair_max_requests_per_iteration")
        .map(|s| s.parse::<usize>().unwrap());
    let serve_repair_oversampled_requests_per_iteration = sub_matches
        .value_of("serve_repair_oversampled_requests_per_iteration")
        .map(|s| s.parse::<usize>().unwrap());
    let serve_repair_ancestor_hashes_invalid_respones = sub_matches
        .value_of("serve_repair_ancestor_hashes_invalid_respones")
        .map(|s| s.parse::<bool>().unwrap());

    configure_repair_parameters(
        rpc_endpoint_url,
        RepairParametersConfig {
            serve_repair_max_requests_per_iteration,
            serve_repair_oversampled_requests_per_iteration,
            serve_repair_ancestor_hashes_invalid_respones,
        },
    )
}

pub fn configure_repair_parameters(
    rpc_endpoint_url: &str,
    repair_parameters_config: RepairParametersConfig,
) -> Result<(), String> {
    repair_parameters_config.send(rpc_endpoint_url)
}
