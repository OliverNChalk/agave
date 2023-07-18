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
    std::net::SocketAddr,
};

impl Command for SendDuplicateBlocksConfig {
    const RPC_METHOD: &'static str = "configureSendDuplicateBlocks";
}

impl Command for InvalidateLeaderBlockConfig {
    const RPC_METHOD: &'static str = "configureInvalidateLeaderBlock";
}

pub fn configure_send_duplicate_blocks_enable(rpc_endpoint_url: String) {
    configure_send_duplicate_blocks(
        &rpc_endpoint_url,
        SendDuplicateBlocksConfig {
            num_duplicate_validators: Some(1),
            new_entry_index_from_end: Some(0),
            send_original_after_ms: Some(0),
            send_destinations: None,
        },
    );
}

pub fn configure_send_duplicate_blocks_disable(rpc_endpoint_url: String) {
    configure_send_duplicate_blocks(
        &rpc_endpoint_url,
        SendDuplicateBlocksConfig {
            num_duplicate_validators: None,
            new_entry_index_from_end: None,
            send_original_after_ms: None,
            send_destinations: None,
        },
    );
}

pub fn configure_send_duplicate_blocks_args(
    rpc_endpoint_url: &str,
    sub_matches: &ArgMatches<'_>,
) -> Result<(), String> {
    let num_duplicate_validators = sub_matches
        .value_of("num_duplicate_validators")
        .map(|s| s.parse::<usize>().unwrap());
    let new_entry_index_from_end = sub_matches
        .value_of("new_entry_index_from_end")
        .map(|s| s.parse::<usize>().unwrap());
    let send_original_after_ms = sub_matches
        .value_of("send_original_after_ms")
        .map(|s| s.parse::<u64>().unwrap());
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
            .collect()
        });

    configure_send_duplicate_blocks(
        rpc_endpoint_url,
        SendDuplicateBlocksConfig {
            num_duplicate_validators,
            new_entry_index_from_end,
            send_original_after_ms,
            send_destinations,
        },
    );

    Ok(())
}

pub fn configure_send_duplicate_blocks(
    rpc_endpoint_url: &str,
    send_duplicate_blocks_config: SendDuplicateBlocksConfig,
) {
    send_duplicate_blocks_config.send(rpc_endpoint_url);
}

pub fn configure_invalidate_leader_block_enable(rpc_endpoint_url: String) {
    configure_invalidate_leader_block(
        &rpc_endpoint_url,
        InvalidateLeaderBlockConfig {
            invalidation_kind: Some(InvalidationKind::InvalidFeePayer),
        },
    );
}

pub fn configure_invalidate_leader_block_disable(rpc_endpoint_url: String) {
    configure_invalidate_leader_block(
        &rpc_endpoint_url,
        InvalidateLeaderBlockConfig {
            invalidation_kind: None,
        },
    );
}

pub fn configure_invalidate_leader_block_args(
    rpc_endpoint_url: &str,
    sub_matches: &clap::ArgMatches<'_>,
) -> Result<(), String> {
    let invalidation_kind = match sub_matches.value_of("invalidation_kind") {
        Some(invalidation_kind) => Some(
            serde_json::from_str(&format!(r#""{invalidation_kind}""#)).map_err(|_| {
                format!("Error converting to enum from string: {invalidation_kind}",)
            })?,
        ),
        None => None,
    };

    configure_invalidate_leader_block(
        rpc_endpoint_url,
        InvalidateLeaderBlockConfig { invalidation_kind },
    );

    Ok(())
}

pub fn configure_invalidate_leader_block(
    rpc_endpoint_url: &str,
    invalidate_leader_block_config: InvalidateLeaderBlockConfig,
) {
    invalidate_leader_block_config.send(rpc_endpoint_url);
}
