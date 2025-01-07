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
        adversary_feature_set::flood_unused_port::{
            AdversarialConfig as UnusedPortPacketFloodConfig, FloodConfig, FloodStrategy,
        },
        verify_peer_identifier,
    },
    solana_keypair::Keypair,
    std::str::FromStr,
};

impl Command for UnusedPortPacketFloodConfig {
    const RPC_METHOD: &'static str = "configureUnusedPortPacketFlood";
}

pub fn configure_unused_port_packet_flood_args(
    rpc_endpoint_url: &str,
    sub_matches: &ArgMatches<'_>,
    rpc_adversary_keypair: &Option<Keypair>,
) -> Result<(), String> {
    let flood_config = if let Ok(path) = value_t!(sub_matches, "toml_config", String) {
        let config_data = load_configuration(&path)?;
        let data: UnusedPortPacketFloodConfig = toml::from_str(&config_data)
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

        let port = match flood_strategy {
            Some(FloodStrategy::HardcodedPort) => match value_t!(sub_matches, "port", u16) {
                Ok(val) => Some(val),
                Err(ClapError {
                    kind: ClapErrorKind::ArgumentNotFound,
                    ..
                }) => return Err("Port must be specified for HardcodedPort strategy".to_string()),
                Err(e) => return Err(e.to_string()),
            },
            _ => None,
        };

        let config = flood_strategy.map(|flood_strategy| FloodConfig {
            flood_strategy,
            packets_per_peer_per_iteration,
            iteration_delay_us,
            target,
            port,
        });
        UnusedPortPacketFloodConfig {
            flood_config: config,
        }
    };
    configure_unused_port_packet_flood(rpc_endpoint_url, flood_config, rpc_adversary_keypair)
}

pub fn configure_unused_port_packet_flood(
    rpc_endpoint_url: &str,
    config: UnusedPortPacketFloodConfig,
    rpc_adversary_keypair: &Option<Keypair>,
) -> Result<(), String> {
    config.send_with_auth(rpc_endpoint_url, rpc_adversary_keypair)
}
