use {
    super::Command,
    clap::{value_t_or_exit, ArgMatches},
    solana_adversary::adversary_feature_set::drop_turbine_votes::AdversarialConfig as DropTurbineVotesConfig,
    solana_keypair::Keypair,
};

impl Command for DropTurbineVotesConfig {
    const RPC_METHOD: &'static str = "turbineVotes";
}

pub fn configure_drop_turbine_votes_enable(
    rpc_endpoint_url: String,
    rpc_adversary_keypair: &Option<Keypair>,
) -> Result<(), String> {
    configure_drop_turbine_votes(
        &rpc_endpoint_url,
        DropTurbineVotesConfig {
            drop_turbine_votes: true,
        },
        rpc_adversary_keypair,
    )
}

pub fn configure_drop_turbine_votes_disable(
    rpc_endpoint_url: String,
    rpc_adversary_keypair: &Option<Keypair>,
) -> Result<(), String> {
    configure_drop_turbine_votes(
        &rpc_endpoint_url,
        DropTurbineVotesConfig {
            drop_turbine_votes: false,
        },
        rpc_adversary_keypair,
    )
}

pub fn configure_drop_turbine_votes_args(
    rpc_endpoint_url: &str,
    sub_matches: &ArgMatches<'_>,
    rpc_adversary_keypair: &Option<Keypair>,
) -> Result<(), String> {
    let drop_turbine_votes = value_t_or_exit!(sub_matches, "drop", bool);

    configure_drop_turbine_votes(
        rpc_endpoint_url,
        DropTurbineVotesConfig { drop_turbine_votes },
        rpc_adversary_keypair,
    )
}

pub fn configure_drop_turbine_votes(
    rpc_endpoint_url: &str,
    drop_turbine_votes_config: DropTurbineVotesConfig,
    rpc_adversary_keypair: &Option<Keypair>,
) -> Result<(), String> {
    drop_turbine_votes_config.send_with_auth(rpc_endpoint_url, rpc_adversary_keypair)
}
