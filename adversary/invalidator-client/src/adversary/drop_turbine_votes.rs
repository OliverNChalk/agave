use {
    super::Command,
    clap::{value_t_or_exit, ArgMatches},
    solana_adversary::adversary_feature_set::drop_turbine_votes::AdversarialConfig as DropTurbineVotesConfig,
};

impl Command for DropTurbineVotesConfig {
    const RPC_METHOD: &'static str = "turbineVotes";
}

pub fn configure_drop_turbine_votes_enable(rpc_endpoint_url: String) {
    configure_drop_turbine_votes(
        &rpc_endpoint_url,
        DropTurbineVotesConfig {
            drop_turbine_votes: true,
        },
    );
}

pub fn configure_drop_turbine_votes_disable(rpc_endpoint_url: String) {
    configure_drop_turbine_votes(
        &rpc_endpoint_url,
        DropTurbineVotesConfig {
            drop_turbine_votes: false,
        },
    );
}

pub fn configure_drop_turbine_votes_args(
    rpc_endpoint_url: &str,
    sub_matches: &ArgMatches<'_>,
) -> Result<(), String> {
    let drop_turbine_votes = value_t_or_exit!(sub_matches, "drop", bool);

    configure_drop_turbine_votes(
        rpc_endpoint_url,
        DropTurbineVotesConfig { drop_turbine_votes },
    );

    Ok(())
}

pub fn configure_drop_turbine_votes(
    rpc_endpoint_url: &str,
    drop_turbine_votes_config: DropTurbineVotesConfig,
) {
    drop_turbine_votes_config.send(rpc_endpoint_url);
}
