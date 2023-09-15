use {
    super::Command,
    crate::{
        adversary::trace,
        common::{
            load_configuration, DEFAULT_FLOOD_ITERATION_DELAY_US,
            DEFAULT_FLOOD_PACKETS_PER_PEER_PER_ITERATION,
        },
    },
    clap::{value_t, ArgMatches, Error as ClapError, ErrorKind as ClapErrorKind},
    solana_adversary::{
        adversary_feature_set::{
            repair_packet_flood::{
                AdversarialConfig as RepairPacketFloodConfig, FloodConfig, FloodStrategy,
            },
            repair_parameters::AdversarialConfig as RepairParametersConfig,
        },
        verify_peer_identifier,
    },
    solana_keypair::Keypair,
    std::str::FromStr,
};

impl Command for RepairPacketFloodConfig {
    const RPC_METHOD: &'static str = "configureRepairPacketFlood";
}

impl Command for RepairParametersConfig {
    const RPC_METHOD: &'static str = "configureRepairParameters";
}

pub fn configure_repair_packet_flood_enable(
    rpc_endpoint_url: String,
    rpc_adversary_keypair: &Option<Keypair>,
) -> Result<(), String> {
    configure_repair_packet_flood(
        &rpc_endpoint_url,
        RepairPacketFloodConfig {
            configs: vec![FloodConfig {
                flood_strategy: FloodStrategy::MinimalPackets,
                packets_per_peer_per_iteration: DEFAULT_FLOOD_PACKETS_PER_PEER_PER_ITERATION,
                iteration_delay_us: DEFAULT_FLOOD_ITERATION_DELAY_US,
                target: None,
            }],
        },
        rpc_adversary_keypair,
    )
}

pub fn configure_repair_packet_flood_disable(
    rpc_endpoint_url: String,
    rpc_adversary_keypair: &Option<Keypair>,
) -> Result<(), String> {
    configure_repair_packet_flood(
        &rpc_endpoint_url,
        RepairPacketFloodConfig { configs: vec![] },
        rpc_adversary_keypair,
    )
}

pub fn configure_repair_packet_flood_args(
    rpc_endpoint_url: &str,
    sub_matches: &ArgMatches<'_>,
    rpc_adversary_keypair: &Option<Keypair>,
) -> Result<(), String> {
    let flood_config = if let Ok(path) = value_t!(sub_matches, "toml_config", String) {
        let config_data = load_configuration(&path)?;
        let data: RepairPacketFloodConfig = toml::from_str(&config_data)
            .map_err(|e| format!("Failed to parse TOML configuration from {path}: {e}"))?;
        trace!("Parsed config:\n{data:#?}");
        data
    } else {
        let flood_strategy = match value_t!(sub_matches, "flood_strategy", String) {
            Ok(val) => {
                let flood_strategy = FloodStrategy::from_str(&val)
                    .map_err(|e| format!("Error converting to enum from string: {e}"))?;
                Some(flood_strategy)
            }
            Err(ClapError {
                kind: ClapErrorKind::ArgumentNotFound,
                ..
            }) => None,
            Err(e) => Err(e.to_string())?,
        };

        let packets_per_peer_per_iteration =
            match value_t!(sub_matches, "packets_per_peer_per_iteration", u32) {
                Ok(val) => val,
                Err(ClapError {
                    kind: ClapErrorKind::ArgumentNotFound,
                    ..
                }) => DEFAULT_FLOOD_PACKETS_PER_PEER_PER_ITERATION,
                Err(e) => Err(e.to_string())?,
            };

        let iteration_delay_us = match value_t!(sub_matches, "iteration_delay_us", u64) {
            Ok(val) => val,
            Err(ClapError {
                kind: ClapErrorKind::ArgumentNotFound,
                ..
            }) => DEFAULT_FLOOD_ITERATION_DELAY_US,
            Err(e) => Err(e.to_string())?,
        };

        let target = sub_matches.value_of("target").map(|s| s.to_string());
        if let Some(ref target) = target {
            verify_peer_identifier(target)?;
        }

        let configs = if let Some(flood_strategy) = flood_strategy {
            vec![FloodConfig {
                flood_strategy,
                packets_per_peer_per_iteration,
                iteration_delay_us,
                target,
            }]
        } else {
            vec![]
        };
        RepairPacketFloodConfig { configs }
    };
    configure_repair_packet_flood(rpc_endpoint_url, flood_config, rpc_adversary_keypair)
}

pub fn configure_repair_packet_flood(
    rpc_endpoint_url: &str,
    repair_packet_flood_config: RepairPacketFloodConfig,
    rpc_adversary_keypair: &Option<Keypair>,
) -> Result<(), String> {
    repair_packet_flood_config.send_with_auth(rpc_endpoint_url, rpc_adversary_keypair)
}

pub fn configure_repair_parameters_args(
    rpc_endpoint_url: &str,
    sub_matches: &ArgMatches<'_>,
    rpc_adversary_keypair: &Option<Keypair>,
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
    let ancestor_hash_repair_sample_size = sub_matches
        .value_of("ancestor_hash_repair_sample_size")
        .map(|s| s.parse::<usize>().unwrap());

    configure_repair_parameters(
        rpc_endpoint_url,
        RepairParametersConfig {
            serve_repair_max_requests_per_iteration,
            serve_repair_oversampled_requests_per_iteration,
            serve_repair_ancestor_hashes_invalid_respones,
            ancestor_hash_repair_sample_size,
        },
        rpc_adversary_keypair,
    )
}

pub fn configure_repair_parameters(
    rpc_endpoint_url: &str,
    repair_parameters_config: RepairParametersConfig,
    rpc_adversary_keypair: &Option<Keypair>,
) -> Result<(), String> {
    repair_parameters_config.send_with_auth(rpc_endpoint_url, rpc_adversary_keypair)
}
