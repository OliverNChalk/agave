use {
    super::Command,
    clap::ArgMatches,
    solana_adversary::adversary_feature_set::replay_stage_attack::{
        AdversarialConfig as ReplayStageAttackConfig, Attack,
    },
    std::str::FromStr,
};

impl Command for ReplayStageAttackConfig {
    const RPC_METHOD: &'static str = "configureReplayStageAttack";
}

pub fn configure_replay_stage_attack_enable(rpc_endpoint_url: String) -> Result<(), String> {
    configure_replay_stage_attack(
        &rpc_endpoint_url,
        ReplayStageAttackConfig {
            selected_attack: Some(Attack::TransferRandom),
        },
    )
}

pub fn configure_replay_stage_attack_disable(rpc_endpoint_url: String) -> Result<(), String> {
    configure_replay_stage_attack(
        &rpc_endpoint_url,
        ReplayStageAttackConfig {
            selected_attack: None,
        },
    )
}

pub fn parse_replay_stage_attack_args(
    sub_matches: &ArgMatches<'_>,
) -> Result<Option<Attack>, String> {
    let Some(selected_attack) = sub_matches.value_of("selected_attack") else {
        return Ok(None);
    };
    let selected_attack = Some(
        Attack::from_str(selected_attack)
            .map_err(|_| format!("Error converting to enum from string: {selected_attack}"))?,
    );
    Ok(selected_attack)
}

pub fn configure_replay_stage_attack_args(
    rpc_endpoint_url: &str,
    sub_matches: &ArgMatches<'_>,
) -> Result<(), String> {
    let selected_attack = parse_replay_stage_attack_args(sub_matches)?;

    configure_replay_stage_attack(
        rpc_endpoint_url,
        ReplayStageAttackConfig { selected_attack },
    )
}

pub fn configure_replay_stage_attack(
    rpc_endpoint_url: &str,
    replay_stage_attack_config: ReplayStageAttackConfig,
) -> Result<(), String> {
    replay_stage_attack_config.send(rpc_endpoint_url)
}
