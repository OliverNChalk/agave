use {
    super::Command, clap::ArgMatches,
    solana_adversary::adversary_feature_set::packet_drop_parameters::AdversarialConfig as PacketDropParametersConfig,
};

impl Command for PacketDropParametersConfig {
    const RPC_METHOD: &'static str = "configurePacketDropParameters";
}

pub fn configure_packet_drop_parameters_args(
    rpc_endpoint_url: &str,
    sub_matches: &ArgMatches<'_>,
) -> Result<(), String> {
    let broadcast_packet_drop_percent = sub_matches
        .value_of("broadcast_packet_drop_percent")
        .map(|s| s.parse::<u8>().unwrap());
    let retransmit_packet_drop_percent = sub_matches
        .value_of("retransmit_packet_drop_percent")
        .map(|s| s.parse::<u8>().unwrap());

    configure_packet_drop_parameters(
        rpc_endpoint_url,
        PacketDropParametersConfig {
            broadcast_packet_drop_percent,
            retransmit_packet_drop_percent,
        },
    )
}

pub fn configure_packet_drop_parameters(
    rpc_endpoint_url: &str,
    packet_drop_parameters_config: PacketDropParametersConfig,
) -> Result<(), String> {
    packet_drop_parameters_config.send(rpc_endpoint_url)
}
