use {
    super::Command, clap::ArgMatches,
    solana_adversary::adversary_feature_set::shred_receiver_address::AdversarialConfig as ShredReceiverAddressConfig,
    solana_keypair::Keypair, std::net::SocketAddr,
};

impl Command for ShredReceiverAddressConfig {
    const RPC_METHOD: &'static str = "configureShredReceiverAddress";
}

pub fn configure_shred_receiver_address_args(
    rpc_endpoint_url: &str,
    sub_matches: &ArgMatches<'_>,
    rpc_adversary_keypair: &Option<Keypair>,
) -> Result<(), String> {
    let shred_receiver_address = sub_matches
        .value_of("shred-receiver-address")
        .map(|s| s.parse::<SocketAddr>().unwrap());

    configure_shred_receiver_address(
        rpc_endpoint_url,
        ShredReceiverAddressConfig {
            shred_receiver_address,
        },
        rpc_adversary_keypair,
    )
}

pub fn configure_shred_receiver_address(
    rpc_endpoint_url: &str,
    shred_receiver_address_config: ShredReceiverAddressConfig,
    rpc_adversary_keypair: &Option<Keypair>,
) -> Result<(), String> {
    shred_receiver_address_config.send_with_auth(rpc_endpoint_url, rpc_adversary_keypair)
}
