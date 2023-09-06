use {
    super::Command,
    clap::ArgMatches,
    log::*,
    solana_adversary::adversary_feature_set::{
        invalidate_leader_block::{
            AdversarialConfig as InvalidateLeaderBlockConfig, InvalidationKind,
        },
        send_duplicate_blocks::AdversarialConfig as SendDuplicateBlocksConfig,
    },
    std::{net::SocketAddr, str::FromStr, sync::Arc},
};

impl Command for SendDuplicateBlocksConfig {
    const RPC_METHOD: &'static str = "configureSendDuplicateBlocks";
}

impl Command for InvalidateLeaderBlockConfig {
    const RPC_METHOD: &'static str = "configureInvalidateLeaderBlock";
}

pub fn configure_send_duplicate_blocks_enable(rpc_endpoint_url: String) -> Result<(), String> {
    configure_send_duplicate_blocks(
        &rpc_endpoint_url,
        SendDuplicateBlocksConfig {
            num_duplicate_validators: 1,
            new_entry_index_from_end: 0,
            send_original_after_ms: 0,
            send_destinations: Vec::new(),
        },
    )
}

pub fn configure_send_duplicate_blocks_disable(rpc_endpoint_url: String) -> Result<(), String> {
    configure_send_duplicate_blocks(&rpc_endpoint_url, SendDuplicateBlocksConfig::default())
}

pub fn configure_send_duplicate_blocks_args(
    rpc_endpoint_url: &str,
    sub_matches: &ArgMatches<'_>,
) -> Result<(), String> {
    let num_duplicate_validators = sub_matches
        .value_of("num_duplicate_validators")
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or_default();
    let new_entry_index_from_end = sub_matches
        .value_of("new_entry_index_from_end")
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or_default();
    let send_original_after_ms = sub_matches
        .value_of("send_original_after_ms")
        .map(|s| s.parse::<u64>().unwrap())
        .unwrap_or_default();
    let send_destinations = sub_matches
        .value_of("send_destinations")
        .map(|send_destinations| {
            // Only allow configuring single partition for now
            vec![send_destinations
                .split(',')
                .filter_map(|s| match s.to_string().parse() {
                    Ok(socket_addr) => Some(socket_addr),
                    Err(_) => {
                        error!("Error parsing socket address: {s:?}");
                        None
                    }
                })
                .collect::<Vec<SocketAddr>>()]
            .into_iter()
            .filter(|partition| !partition.is_empty())
            .map(Arc::new)
            .collect()
        })
        .unwrap_or_default();

    configure_send_duplicate_blocks(
        rpc_endpoint_url,
        SendDuplicateBlocksConfig {
            num_duplicate_validators,
            new_entry_index_from_end,
            send_original_after_ms,
            send_destinations,
        },
    )
}

pub fn configure_send_duplicate_blocks(
    rpc_endpoint_url: &str,
    send_duplicate_blocks_config: SendDuplicateBlocksConfig,
) -> Result<(), String> {
    send_duplicate_blocks_config.send(rpc_endpoint_url)
}

pub fn configure_invalidate_leader_block_enable(rpc_endpoint_url: String) -> Result<(), String> {
    configure_invalidate_leader_block(
        &rpc_endpoint_url,
        InvalidateLeaderBlockConfig {
            invalidation_kind: Some(InvalidationKind::InvalidFeePayer),
        },
    )
}

pub fn configure_invalidate_leader_block_disable(rpc_endpoint_url: String) -> Result<(), String> {
    configure_invalidate_leader_block(
        &rpc_endpoint_url,
        InvalidateLeaderBlockConfig {
            invalidation_kind: None,
        },
    )
}

pub fn configure_invalidate_leader_block_args(
    rpc_endpoint_url: &str,
    sub_matches: &clap::ArgMatches<'_>,
) -> Result<(), String> {
    let invalidation_kind = match sub_matches.value_of("invalidation_kind") {
        Some(invalidation_kind) => {
            Some(InvalidationKind::from_str(invalidation_kind).map_err(|_| {
                format!("Error converting to enum from string: {invalidation_kind}")
            })?)
        }
        None => None,
    };

    configure_invalidate_leader_block(
        rpc_endpoint_url,
        InvalidateLeaderBlockConfig { invalidation_kind },
    )
}

pub fn configure_invalidate_leader_block(
    rpc_endpoint_url: &str,
    invalidate_leader_block_config: InvalidateLeaderBlockConfig,
) -> Result<(), String> {
    invalidate_leader_block_config.send(rpc_endpoint_url)
}
