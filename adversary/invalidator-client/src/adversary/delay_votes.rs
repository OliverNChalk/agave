use {
    super::Command,
    clap::{value_t_or_exit, ArgMatches},
    solana_adversary::adversary_feature_set::delay_votes::AdversarialConfig as DelayVotesConfig,
    solana_keypair::Keypair,
};

impl Command for DelayVotesConfig {
    const RPC_METHOD: &'static str = "delayVotes";
}

pub fn configure_delay_votes_args(
    rpc_endpoint_url: &str,
    sub_matches: &ArgMatches<'_>,
    rpc_adversary_keypair: &Option<Keypair>,
) -> Result<(), String> {
    let delay_votes_by_slot_count = value_t_or_exit!(sub_matches, "slot_count", u64);

    configure_delay_votes(
        rpc_endpoint_url,
        DelayVotesConfig {
            delay_votes_by_slot_count,
        },
        rpc_adversary_keypair,
    )
}

pub fn configure_delay_votes(
    rpc_endpoint_url: &str,
    delay_votes_config: DelayVotesConfig,
    rpc_adversary_keypair: &Option<Keypair>,
) -> Result<(), String> {
    delay_votes_config.send_with_auth(rpc_endpoint_url, rpc_adversary_keypair)
}
