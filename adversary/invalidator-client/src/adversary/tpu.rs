use {
    super::Command,
    crate::{
        adversary::trace,
        common::{
            load_configuration, DEFAULT_FLOOD_ITERATION_DELAY_US,
            DEFAULT_FLOOD_PACKETS_PER_PEER_PER_ITERATION,
        },
    },
    clap::{value_t, value_t_or_exit, ArgMatches, Error as ClapError, ErrorKind as ClapErrorKind},
    solana_adversary::adversary_feature_set::tpu_packet_flood::{
        AdversarialConfig as TpuPacketFloodConfig, FloodConfig, FloodStrategy,
    },
    solana_keypair::Keypair,
    solana_pubkey::Pubkey,
    std::{str::FromStr, time::Duration},
};

impl Command for TpuPacketFloodConfig {
    const RPC_METHOD: &'static str = "configureTpuPacketFlood";
}

pub fn configure_tpu_packet_flood_args(
    rpc_endpoint_url: &str,
    sub_matches: &ArgMatches<'_>,
    rpc_adversary_keypair: &Option<Keypair>,
) -> Result<(), String> {
    let flood_config = if let Ok(path) = value_t!(sub_matches, "toml_config", String) {
        let config_data = load_configuration(&path)?;
        let data: TpuPacketFloodConfig = toml::from_str(&config_data)
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
            match value_t!(sub_matches, "packets_per_peer_per_iteration", usize) {
                Ok(val) => val,
                Err(ClapError {
                    kind: ClapErrorKind::ArgumentNotFound,
                    ..
                }) => DEFAULT_FLOOD_PACKETS_PER_PEER_PER_ITERATION as usize,
                Err(e) => Err(e.to_string())?,
            };
        let iteration_duration = match value_t!(sub_matches, "iteration_duration_us", u64) {
            Ok(val) => Duration::from_micros(val),
            Err(ClapError {
                kind: ClapErrorKind::ArgumentNotFound,
                ..
            }) => Duration::from_micros(DEFAULT_FLOOD_ITERATION_DELAY_US),
            Err(e) => Err(e.to_string())?,
        };
        let target = match value_t!(sub_matches, "target", Pubkey) {
            Ok(val) => Some(val),
            Err(ClapError {
                kind: ClapErrorKind::ArgumentNotFound,
                ..
            }) => None,
            Err(e) => Err(e.to_string())?,
        };
        let target_leader = value_t_or_exit!(sub_matches, "target_leader", bool);
        let configs = if let Some(flood_strategy) = flood_strategy {
            vec![FloodConfig {
                flood_strategy,
                packets_per_peer_per_iteration,
                iteration_duration,
                target,
                target_leader,
            }]
        } else {
            vec![]
        };
        TpuPacketFloodConfig { configs }
    };
    configure_tpu_packet_flood(rpc_endpoint_url, flood_config, rpc_adversary_keypair)
}

pub fn configure_tpu_packet_flood(
    rpc_endpoint_url: &str,
    config: TpuPacketFloodConfig,
    rpc_adversary_keypair: &Option<Keypair>,
) -> Result<(), String> {
    config.send_with_auth(rpc_endpoint_url, rpc_adversary_keypair)
}
