use {
    crate::common::STDIN_TOKEN,
    clap::{App, AppSettings, Arg, SubCommand},
    const_format::formatcp,
    solana_adversary::{
        adversary_feature_set::{
            gossip_packet_flood::FloodStrategy as GossipFloodStrategy,
            invalidate_leader_block::InvalidationKind,
            repair_packet_flood::FloodStrategy as RepairFloodStrategy,
            replay_stage_attack::Attack as ReplayStageAttack,
            tpu_packet_flood::FloodStrategy as TpuFloodStrategy,
        },
        tpu::MAX_PACKETS_PER_PEER_PER_ITERATION,
    },
    solana_clap_utils::{
        input_parsers::keypair_of,
        input_validators::{is_keypair_or_ask_keyword, is_url_or_moniker},
        *,
    },
    solana_cli_config::ConfigInput,
};

const RPC_ENDPOINT_URL: &str = "http://localhost:8899";

fn build_args<'a>(version: &'static str) -> App<'a, 'static> {
    // to fix error inside formatcp macro
    #![allow(clippy::arithmetic_side_effects)]
    App::new("InvalidatorClient")
        .version(version)
        .about("Client for interacting with the Solana Invalidator")
        .arg(
            Arg::with_name("json_rpc_url")
                .short("u")
                .long("url")
                .value_name("URL_OR_MONIKER")
                .takes_value(true)
                .global(true)
                .validator(is_url_or_moniker)
                .help(
                    "URL for Solana's JSON RPC or moniker (or their first letter): [mainnet-beta, \
                     testnet, devnet, localhost]",
                ),
        )
        .arg(
            Arg::with_name("rpc_adversary_keypair")
                .long("rpc-adversary-keypair")
                .value_name("KEYPAIR")
                .takes_value(true)
                .validator(is_keypair_or_ask_keyword)
                .help("Validator rpc adversary keypair"),
        )
        .subcommand(
            SubCommand::with_name("configure-shred-receiver-address")
                .about("Configure the shred receiver address")
                .arg(
                    Arg::with_name("shred-receiver-address")
                        .long("shred-receiver-address")
                        .takes_value(true)
                        .value_name("SOCKET ADDRESS")
                        .validator(solana_net_utils::is_host_port)
                        .help("Address to forward shreds to"),
                ),
        )
        .subcommand(
            SubCommand::with_name("configure-repair-packet-flood")
                .about("Configure flooding repair packet requests")
                .arg(
                    Arg::with_name("flood_strategy")
                        .long("flood-strategy")
                        .value_name("ENUM STRING")
                        .possible_values(RepairFloodStrategy::cli_names())
                        .help("Which strategy to use for flooding repair packets")
                        .conflicts_with("toml_config"),
                )
                .arg(
                    Arg::with_name("packets_per_peer_per_iteration")
                        .long("packets-per-peer-per-iteration")
                        .value_name("NUMBER")
                        .validator(input_validators::is_parsable::<u32>)
                        .help("Number of packets to send to each peer each iteration")
                        .conflicts_with("toml_config")
                        .requires("flood_strategy"),
                )
                .arg(
                    Arg::with_name("iteration_delay_us")
                        .long("iteration-delay-us")
                        .value_name("MICROSECONDS")
                        .validator(input_validators::is_parsable::<u64>)
                        .help("Delay between iterations in microseconds")
                        .conflicts_with("toml_config")
                        .requires("flood_strategy"),
                )
                .arg(
                    Arg::with_name("target")
                        .long("target")
                        .takes_value(true)
                        .value_name("PUBKEY")
                        .validator(input_validators::is_pubkey)
                        .help("Peer to target with repair packets")
                        .conflicts_with("toml_config")
                        .requires("flood_strategy"),
                )
                .arg(
                    Arg::with_name("toml_config")
                        .long("toml")
                        .takes_value(true)
                        .value_name("FILE")
                        .help(formatcp!(
                            "TOML input file path or \"{STDIN_TOKEN}\" for stdin."
                        ))
                        .conflicts_with("flood_strategy"),
                ),
        )
        .subcommand(
            SubCommand::with_name("configure-repair-parameters")
                .about("Configure the repair parameters")
                .arg(
                    Arg::with_name("serve_repair_max_requests_per_iteration")
                        .long("serve-repair-max-requests-per-iteration")
                        .takes_value(true)
                        .value_name("NUMBER")
                        .validator(input_validators::is_parsable::<u64>)
                        .help("Maximum number of repair requests to serve per iteration"),
                )
                .arg(
                    Arg::with_name("serve_repair_oversampled_requests_per_iteration")
                        .long("serve-repair-oversampled-requests-per-iteration")
                        .takes_value(true)
                        .value_name("NUMBER")
                        .validator(input_validators::is_parsable::<u64>)
                        .help("Oversampled requests to serve per iteration"),
                )
                .arg(
                    Arg::with_name("serve_repair_ancestor_hashes_invalid_respones")
                        .long("serve-repair-ancestor-hashes-invalid-respones")
                        .default_value("false")
                        .takes_value(true)
                        .value_name("BOOLEAN")
                        .validator(input_validators::is_parsable::<bool>)
                        .help("Return invalid ancestor hashes values"),
                )
                .arg(
                    Arg::with_name("ancestor_hash_repair_sample_size")
                        .long("ancestor-hash-repair-sample-size")
                        .takes_value(true)
                        .value_name("NUMBER")
                        .validator(input_validators::is_parsable::<u64>)
                        .help("Override ancestor hash repair sample size"),
                ),
        )
        .subcommand(
            SubCommand::with_name("configure-send-duplicate-blocks")
                .about(
                    "Configure sending duplicate leader blocks. Requires leader slots to do \
                     anything.",
                )
                .arg(
                    Arg::with_name("num_duplicate_validators")
                        .long("num-duplicate-validators")
                        .takes_value(true)
                        .value_name("NUMBER")
                        .validator(|arg| input_validators::is_within_range(arg, 1..))
                        .help("How many duplicate blocks to generate"),
                )
                .arg(
                    Arg::with_name("new_entry_index_from_end")
                        .long("new-entry-index-from-end")
                        .takes_value(true)
                        .value_name("INDEX")
                        .validator(input_validators::is_parsable::<usize>)
                        .help("Entry index to remove from the end of the block"),
                )
                .arg(
                    Arg::with_name("send_original_after_ms")
                        .long("send-original-after-ms")
                        .takes_value(true)
                        .value_name("MILLISECONDS")
                        .validator(input_validators::is_parsable::<u64>)
                        .help(
                            "Delay between sending the duplicate and original block in \
                             milliseconds",
                        ),
                )
                .arg(
                    Arg::with_name("turbine_send_delay_ms")
                        .long("turbine-send-delay-ms")
                        .takes_value(true)
                        .value_name("MILLISECONDS")
                        .validator(input_validators::is_parsable::<u64>)
                        .help("Delay before broadcasting block in milliseconds"),
                )
                .arg(
                    Arg::with_name("send_destinations")
                        .long("send-destinations")
                        .takes_value(true)
                        .value_name("SOCKET ADDRESSES")
                        .help("CSV of peer addresses to target with duplicate block"),
                )
                .arg(
                    Arg::with_name("leaf_node_partitions")
                        .long("leaf-node-partitions")
                        .takes_value(true)
                        .value_name("PARTITIONS")
                        .validator(|arg| input_validators::is_within_range(arg, 1..))
                        .help("How many duplicates and leaf node partitions to create"),
                ),
        )
        .subcommand(
            SubCommand::with_name("configure-invalidate-leader-block")
                .about(
                    "Configure invalidating the leader block. Requires leader slots to do \
                     anything.",
                )
                .arg(
                    Arg::with_name("invalidation_kind")
                        .long("invalidation-kind")
                        .takes_value(true)
                        .value_name("ENUM STRING")
                        .possible_values(InvalidationKind::cli_names())
                        .help("Manner in which to invalidate the leader block"),
                ),
        )
        .subcommand(
            SubCommand::with_name("configure-drop-turbine-votes")
                .about("Configure dropping votes received from turbine")
                .arg(
                    Arg::with_name("drop")
                        .long("drop")
                        .default_value("true")
                        .value_name("BOOLEAN")
                        .validator(input_validators::is_parsable::<bool>)
                        .help(
                            "Drop all votes received from turbine. Don't forward or include in \
                             leader block.",
                        ),
                ),
        )
        .subcommand(
            SubCommand::with_name("configure-delay-votes")
                .about("Configure delaying sending votes to leader")
                .arg(
                    Arg::with_name("slot_count")
                        .long("slot-count")
                        .default_value("0")
                        .value_name("NUMBER")
                        .validator(input_validators::is_parsable::<u64>)
                        .help("Number of slots to delay sending votes by."),
                ),
        )
        .subcommand(
            SubCommand::with_name("configure-packet-drop-parameters")
                .about("Configure parameters to control dropping packets")
                .arg(
                    Arg::with_name("broadcast_packet_drop_percent")
                        .long("broadcast-packet-drop-percent")
                        .takes_value(true)
                        .value_name("NUMBER")
                        .validator(input_validators::is_valid_percentage)
                        .help("Percent of outgoing broadcast packets to drop"),
                )
                .arg(
                    Arg::with_name("retransmit_packet_drop_percent")
                        .long("retransmit-packet-drop-percent")
                        .takes_value(true)
                        .value_name("NUMBER")
                        .validator(input_validators::is_valid_percentage)
                        .help("Percent of outgoing retransmit packets to drop"),
                ),
        )
        .subcommand(
            SubCommand::with_name("configure-gossip-packet-flood")
                .about("Configure flooding gossip packet requests")
                .arg(
                    Arg::with_name("flood_strategy")
                        .long("flood-strategy")
                        .value_name("ENUM STRING")
                        .possible_values(GossipFloodStrategy::cli_names())
                        .help("Which strategy to use for flooding gossip packets")
                        .conflicts_with("toml_config"),
                )
                .arg(
                    Arg::with_name("packets_per_peer_per_iteration")
                        .long("packets-per-peer-per-iteration")
                        .value_name("NUMBER")
                        .validator(input_validators::is_parsable::<u32>)
                        .help("Number of packets to send to each peer each iteration")
                        .conflicts_with("toml_config"),
                )
                .arg(
                    Arg::with_name("iteration_delay_us")
                        .long("iteration-delay-us")
                        .value_name("MICROSECONDS")
                        .validator(input_validators::is_parsable::<u64>)
                        .help("Delay between iterations in microseconds")
                        .conflicts_with("toml_config"),
                )
                .arg(
                    Arg::with_name("target")
                        .long("target")
                        .takes_value(true)
                        .value_name("PUBKEY")
                        .validator(input_validators::is_pubkey)
                        .help("Peer to target with gossip packets")
                        .conflicts_with("toml_config"),
                )
                .arg(
                    Arg::with_name("toml_config")
                        .long("toml")
                        .takes_value(true)
                        .value_name("FILE")
                        .help(formatcp!(
                            "TOML input file path or \"{STDIN_TOKEN}\" for stdin."
                        )),
                ),
        )
        .subcommand(
            SubCommand::with_name("configure-tpu-packet-flood")
                .about("Configure flooding TPU packet requests")
                .arg(
                    Arg::with_name("flood_strategy")
                        .long("flood-strategy")
                        .value_name("ENUM STRING")
                        .possible_values(TpuFloodStrategy::cli_names())
                        .help("Which strategy to use for flooding TPU packets")
                        .conflicts_with("toml_config"),
                )
                .arg(
                    Arg::with_name("packets_per_peer_per_iteration")
                        .long("packets-per-peer-per-iteration")
                        .value_name("NUMBER")
                        .validator(|arg| {
                            input_validators::is_within_range(
                                arg,
                                1..MAX_PACKETS_PER_PEER_PER_ITERATION,
                            )
                        })
                        .help("Number of packets to send to each peer each iteration")
                        .conflicts_with("toml_config"),
                )
                .arg(
                    Arg::with_name("iteration_duration_us")
                        .long("iteration-duration-us")
                        .value_name("MICROSECONDS")
                        .validator(input_validators::is_parsable::<u64>)
                        .help("Minimum time for an iteration in microseconds")
                        .conflicts_with("toml_config"),
                )
                .arg(
                    Arg::with_name("target")
                        .long("target")
                        .takes_value(true)
                        .value_name("PUBKEY")
                        .validator(input_validators::is_pubkey)
                        .help("Peer to target with packets")
                        .conflicts_with("target_leader")
                        .conflicts_with("toml_config"),
                )
                .arg(
                    Arg::with_name("target_leader")
                        .long("target-leader")
                        .default_value("true")
                        .value_name("BOOLEAN")
                        .validator(input_validators::is_parsable::<bool>)
                        .help("Target current leader with packets")
                        .conflicts_with("target")
                        .conflicts_with("toml_config"),
                )
                .arg(
                    Arg::with_name("toml_config")
                        .long("toml")
                        .takes_value(true)
                        .value_name("FILE")
                        .help(formatcp!(
                            "TOML input file path or \"{STDIN_TOKEN}\" for stdin."
                        )),
                ),
        )
        .subcommand(
            SubCommand::with_name("configure-replay-stage-attack")
                .about(
                    "Configure packing custom blocks to replay. Requires leader slots to do \
                     anything.",
                )
                .arg(
                    Arg::with_name("selected_attack")
                        .long("selected-attack")
                        .takes_value(true)
                        .value_name("ENUM STRING")
                        .possible_values(ReplayStageAttack::cli_names())
                        .help("Which replay attack to perform"),
                )
                .arg(
                    Arg::with_name("transaction_batch_size")
                        .long("transaction-batch-size")
                        .takes_value(true)
                        .value_name("NUMBER")
                        .validator(input_validators::is_parsable::<usize>)
                        .help("Number of transactions in a batch (entry) for replay execution."),
                )
                .arg(
                    Arg::with_name("num_accounts_per_tx")
                        .long("num-accounts-per-tx")
                        .value_name("NUMBER")
                        .validator(input_validators::is_parsable::<usize>)
                        .default_value("1")
                        .help("Number of accounts a transaction access."),
                )
                .arg(
                    Arg::with_name("transaction_cu_budget")
                        .long("transaction-cu-budget")
                        .value_name("NUMBER")
                        .validator(input_validators::is_parsable::<u32>)
                        .help(
                            "CU budget for a transaction. Setting this to lower than required \
                             value might be used to make transaction to be invalid.",
                        ),
                )
                .arg(
                    Arg::with_name("use_failed_transaction_hotpath")
                        .long("use-failed-transaction-hotpath")
                        .takes_value(false)
                        .help(
                            "Enable hotpath to skip the execution on the invalidator node. This \
                             requires invalid transactions which can be achieved by using too \
                             small CU budget.",
                        ),
                )
                .arg(
                    Arg::with_name("tx_data_size")
                        .long("tx-data-size")
                        .value_name("NUMBER")
                        .validator(input_validators::is_parsable::<usize>)
                        .help(
                            "Amount of padding to add to large nop transactions. Default ensures \
                             max tx size.",
                        ),
                )
                .arg(
                    Arg::with_name("use_invalid_fee_payer")
                        .long("use-invalid-fee-payer")
                        .takes_value(false)
                        .help(
                            "Uses an invalid fee payer account for transactions.  Applies only to \
                             the 'read_non_existent_accounts' attack.",
                        ),
                ),
        )
        .setting(AppSettings::SubcommandRequiredElseHelp)
}

pub fn run_command() -> Result<(), String> {
    let matches = build_args("1.0.0").get_matches();

    let (_, rpc_endpoint_url) = ConfigInput::compute_json_rpc_url_setting(
        matches.value_of("json_rpc_url").unwrap_or(""),
        RPC_ENDPOINT_URL,
    );

    let rpc_adversary_keypair = keypair_of(&matches, "rpc_adversary_keypair");

    match matches.subcommand() {
        ("configure-shred-receiver-address", Some(sub_matches)) => {
            crate::adversary::shred_forwarder::configure_shred_receiver_address_args(
                &rpc_endpoint_url,
                sub_matches,
                &rpc_adversary_keypair,
            )
        }
        ("configure-repair-packet-flood", Some(sub_matches)) => {
            crate::adversary::repair::configure_repair_packet_flood_args(
                &rpc_endpoint_url,
                sub_matches,
                &rpc_adversary_keypair,
            )
        }
        ("configure-repair-parameters", Some(sub_matches)) => {
            crate::adversary::repair::configure_repair_parameters_args(
                &rpc_endpoint_url,
                sub_matches,
                &rpc_adversary_keypair,
            )
        }
        ("configure-send-duplicate-blocks", Some(sub_matches)) => {
            crate::adversary::leader_block::configure_send_duplicate_blocks_args(
                &rpc_endpoint_url,
                sub_matches,
                &rpc_adversary_keypair,
            )
        }
        ("configure-invalidate-leader-block", Some(sub_matches)) => {
            crate::adversary::leader_block::configure_invalidate_leader_block_args(
                &rpc_endpoint_url,
                sub_matches,
                &rpc_adversary_keypair,
            )
        }
        ("configure-delay-votes", Some(sub_matches)) => {
            crate::adversary::delay_votes::configure_delay_votes_args(
                &rpc_endpoint_url,
                sub_matches,
                &rpc_adversary_keypair,
            )
        }
        ("configure-drop-turbine-votes", Some(sub_matches)) => {
            crate::adversary::drop_turbine_votes::configure_drop_turbine_votes_args(
                &rpc_endpoint_url,
                sub_matches,
                &rpc_adversary_keypair,
            )
        }
        ("configure-packet-drop-parameters", Some(sub_matches)) => {
            crate::adversary::packet_drop::configure_packet_drop_parameters_args(
                &rpc_endpoint_url,
                sub_matches,
                &rpc_adversary_keypair,
            )
        }
        ("configure-gossip-packet-flood", Some(sub_matches)) => {
            crate::adversary::gossip::configure_gossip_packet_flood_args(
                &rpc_endpoint_url,
                sub_matches,
                &rpc_adversary_keypair,
            )
        }
        ("configure-tpu-packet-flood", Some(sub_matches)) => {
            crate::adversary::tpu::configure_tpu_packet_flood_args(
                &rpc_endpoint_url,
                sub_matches,
                &rpc_adversary_keypair,
            )
        }
        ("configure-replay-stage-attack", Some(sub_matches)) => {
            crate::adversary::replay::configure_replay_stage_attack_args(
                &rpc_endpoint_url,
                sub_matches,
                &rpc_adversary_keypair,
            )
        }
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        crate::adversary::replay::parse_replay_stage_attack_args,
        solana_adversary::adversary_feature_set::replay_stage_attack::{
            Attack, AttackProgramConfig, LargeNopAttackConfig, NonExistentAccountsAttackConfig,
        },
    };

    // Converts CLI arguments of the form
    //
    // ```
    // solana-invalidator-client -u RPC_ENDPOINT_URL configure-replay-stage-attack \
    //     --selected-attack [attack_name_and_extra_args]
    // ```
    //
    // into an [`Attack`] value and verifies that it matches the `expected_attack`.
    #[track_caller]
    fn check_configure_replay_stage_attack_arg_parsing(
        attack_name_and_extra_args: &[&str],
        expected_attack: Attack,
    ) {
        let args = [
            "solana-invalidator-client",
            "-u",
            RPC_ENDPOINT_URL,
            "configure-replay-stage-attack",
            "--selected-attack",
        ]
        .iter()
        .chain(attack_name_and_extra_args.iter())
        .copied()
        .collect::<Vec<_>>();

        let matches = build_args("1.0.0").get_matches_from(args);

        let (sub_command_name, sub_matches) = matches.subcommand();
        assert_eq!(sub_command_name, "configure-replay-stage-attack");

        let sub_matches =
            sub_matches.expect("A subcommand name was provided and it was checked above");

        let actual_attack = parse_replay_stage_attack_args(sub_matches);
        assert_eq!(actual_attack, Ok(Some(expected_attack)));
    }

    #[test]
    fn test_cli_parse_replay_stage_attack_transfer_random() {
        check_configure_replay_stage_attack_arg_parsing(
            &["transferRandom"],
            Attack::TransferRandom,
        );
    }

    #[test]
    fn test_cli_parse_replay_stage_attack_create_nonce_accounts() {
        check_configure_replay_stage_attack_arg_parsing(
            &["createNonceAccounts"],
            Attack::CreateNonceAccounts,
        );
    }

    #[test]
    fn test_cli_parse_replay_stage_attack_allocate_random_large() {
        check_configure_replay_stage_attack_arg_parsing(
            &["allocateRandomLarge"],
            Attack::AllocateRandomLarge,
        );
    }

    #[test]
    fn test_cli_parse_replay_stage_attack_allocate_random_small() {
        check_configure_replay_stage_attack_arg_parsing(
            &["allocateRandomSmall"],
            Attack::AllocateRandomSmall,
        );
    }

    #[test]
    fn test_cli_parse_replay_stage_attack_chain_transactions() {
        check_configure_replay_stage_attack_arg_parsing(
            &["chainTransactions"],
            Attack::ChainTransactions,
        );
    }

    #[test]
    fn test_cli_parse_replay_stage_attack_read_max_accounts() {
        check_configure_replay_stage_attack_arg_parsing(
            &["readMaxSizeAccounts"],
            Attack::ReadMaxSizeAccounts,
        );
    }

    #[test]
    fn test_cli_parse_replay_stage_attack_write_max_accounts() {
        check_configure_replay_stage_attack_arg_parsing(
            &["writeMaxSizeAccounts"],
            Attack::WriteMaxSizeAccounts,
        );
    }
    #[test]
    fn test_cli_parse_replay_stage_attack_large_nop_default_parameters() {
        check_configure_replay_stage_attack_arg_parsing(
            &["largeNop"],
            Attack::LargeNop(LargeNopAttackConfig::default()),
        );
    }

    #[test]
    fn test_cli_parse_replay_stage_attack_large_nop_custom_parameters() {
        check_configure_replay_stage_attack_arg_parsing(
            &[
                "largeNop",
                "--tx-data-size",
                "100",
                "--transaction-batch-size",
                "32",
                "--num-accounts-per-tx",
                "8",
                "--transaction-cu-budget",
                "100",
            ],
            Attack::LargeNop(LargeNopAttackConfig {
                common: AttackProgramConfig {
                    transaction_batch_size: 32,
                    num_accounts_per_tx: 8,
                    transaction_cu_budget: 100,
                    use_failed_transaction_hotpath: false,
                },
                tx_data_size: 100,
            }),
        );
    }

    #[test]
    fn test_cli_parse_replay_stage_attack_transfer_random_with_memo() {
        check_configure_replay_stage_attack_arg_parsing(
            &["transferRandomWithMemo"],
            Attack::TransferRandomWithMemo,
        );
    }

    #[test]
    fn test_cli_parse_replay_stage_attack_write_program_default_parameters() {
        check_configure_replay_stage_attack_arg_parsing(
            &["writeProgram"],
            Attack::WriteProgram(AttackProgramConfig::default()),
        );
    }

    #[test]
    fn test_cli_parse_replay_stage_attack_write_program_custom_parameters() {
        check_configure_replay_stage_attack_arg_parsing(
            &[
                "writeProgram",
                "--transaction-batch-size",
                "32",
                "--num-accounts-per-tx",
                "8",
                "--transaction-cu-budget",
                "100",
            ],
            Attack::WriteProgram(AttackProgramConfig {
                transaction_batch_size: 32,
                num_accounts_per_tx: 8,
                transaction_cu_budget: 100,
                use_failed_transaction_hotpath: false,
            }),
        );
    }

    #[test]
    fn test_cli_parse_replay_stage_attack_write_program_custom_parameters_fail_hotpath() {
        check_configure_replay_stage_attack_arg_parsing(
            &[
                "writeProgram",
                "--use-failed-transaction-hotpath",
                "--transaction-batch-size",
                "32",
                "--num-accounts-per-tx",
                "8",
                "--transaction-cu-budget",
                "100",
            ],
            Attack::WriteProgram(AttackProgramConfig {
                transaction_batch_size: 32,
                num_accounts_per_tx: 8,
                transaction_cu_budget: 100,
                use_failed_transaction_hotpath: true,
            }),
        );
    }

    #[test]
    fn test_cli_parse_replay_stage_attack_read_program_custom_parameters() {
        check_configure_replay_stage_attack_arg_parsing(
            &[
                "readProgram",
                "--transaction-batch-size",
                "61",
                "--num-accounts-per-tx",
                "4",
                "--transaction-cu-budget",
                "2000",
            ],
            Attack::ReadProgram(AttackProgramConfig {
                transaction_batch_size: 61,
                num_accounts_per_tx: 4,
                transaction_cu_budget: 2000,
                use_failed_transaction_hotpath: false,
            }),
        );
    }

    #[test]
    fn test_cli_parse_replay_stage_attack_read_program_custom_parameters_fail_hotpath() {
        check_configure_replay_stage_attack_arg_parsing(
            &[
                "readProgram",
                "--use-failed-transaction-hotpath",
                "--transaction-batch-size",
                "61",
                "--num-accounts-per-tx",
                "4",
                "--transaction-cu-budget",
                "2000",
            ],
            Attack::ReadProgram(AttackProgramConfig {
                transaction_batch_size: 61,
                num_accounts_per_tx: 4,
                transaction_cu_budget: 2000,
                use_failed_transaction_hotpath: true,
            }),
        );
    }

    #[test]
    fn test_cli_parse_replay_stage_attack_recursive_program_custom_parameters() {
        check_configure_replay_stage_attack_arg_parsing(
            &[
                "recursiveProgram",
                "--transaction-batch-size",
                "40",
                "--num-accounts-per-tx",
                "1",
                "--transaction-cu-budget",
                "1000",
            ],
            Attack::RecursiveProgram(AttackProgramConfig {
                transaction_batch_size: 40,
                num_accounts_per_tx: 1,
                transaction_cu_budget: 1000,
                use_failed_transaction_hotpath: false,
            }),
        );
    }

    #[test]
    fn test_cli_parse_replay_stage_attack_recursive_program_custom_parameters_fail_hotpath() {
        check_configure_replay_stage_attack_arg_parsing(
            &[
                "recursiveProgram",
                "--use-failed-transaction-hotpath",
                "--transaction-batch-size",
                "40",
                "--num-accounts-per-tx",
                "1",
                "--transaction-cu-budget",
                "1000",
            ],
            Attack::RecursiveProgram(AttackProgramConfig {
                transaction_batch_size: 40,
                num_accounts_per_tx: 1,
                transaction_cu_budget: 1000,
                use_failed_transaction_hotpath: true,
            }),
        );
    }

    #[test]
    fn test_cli_parse_replay_stage_attack_cpi_program_custom_parameters() {
        check_configure_replay_stage_attack_arg_parsing(
            &[
                "cpiProgram",
                "--transaction-batch-size",
                "40",
                "--num-accounts-per-tx",
                "1",
                "--transaction-cu-budget",
                "1000",
            ],
            Attack::CpiProgram(AttackProgramConfig {
                transaction_batch_size: 40,
                num_accounts_per_tx: 1,
                transaction_cu_budget: 1000,
                use_failed_transaction_hotpath: false,
            }),
        );
    }

    #[test]
    fn test_cli_parse_replay_stage_attack_cpi_program_custom_parameters_fail_hotpath() {
        check_configure_replay_stage_attack_arg_parsing(
            &[
                "cpiProgram",
                "--use-failed-transaction-hotpath",
                "--transaction-batch-size",
                "40",
                "--num-accounts-per-tx",
                "1",
                "--transaction-cu-budget",
                "1000",
            ],
            Attack::CpiProgram(AttackProgramConfig {
                transaction_batch_size: 40,
                num_accounts_per_tx: 1,
                transaction_cu_budget: 1000,
                use_failed_transaction_hotpath: true,
            }),
        );
    }

    #[test]
    fn test_cli_parse_replay_stage_attack_cold_program_cache_custom_parameters() {
        check_configure_replay_stage_attack_arg_parsing(
            &[
                "coldProgramCache",
                "--transaction-batch-size",
                "2",
                "--num-accounts-per-tx",
                "2",
                "--transaction-cu-budget",
                "100",
            ],
            Attack::ColdProgramCache(AttackProgramConfig {
                transaction_batch_size: 2,
                num_accounts_per_tx: 2,
                transaction_cu_budget: 100,
                use_failed_transaction_hotpath: false,
            }),
        );
    }

    #[test]
    fn test_cli_parse_replay_stage_attack_cold_program_cache_custom_parameters_fail_hotpath() {
        check_configure_replay_stage_attack_arg_parsing(
            &[
                "coldProgramCache",
                "--use-failed-transaction-hotpath",
                "--transaction-batch-size",
                "2",
                "--num-accounts-per-tx",
                "2",
                "--transaction-cu-budget",
                "100",
            ],
            Attack::ColdProgramCache(AttackProgramConfig {
                transaction_batch_size: 2,
                num_accounts_per_tx: 2,
                transaction_cu_budget: 100,
                use_failed_transaction_hotpath: true,
            }),
        );
    }

    #[test]
    fn test_cli_parse_replay_stage_attack_non_existent_accounts_default_parameters() {
        check_configure_replay_stage_attack_arg_parsing(
            &["readNonExistentAccounts"],
            Attack::ReadNonExistentAccounts(NonExistentAccountsAttackConfig::default()),
        );
    }

    #[test]
    fn test_cli_parse_replay_stage_attack_non_existent_accounts_invalid_payer_hotpath_parameters() {
        check_configure_replay_stage_attack_arg_parsing(
            &[
                "readNonExistentAccounts",
                "--use-failed-transaction-hotpath",
                "--use-invalid-fee-payer",
            ],
            Attack::ReadNonExistentAccounts(NonExistentAccountsAttackConfig {
                use_failed_transaction_hotpath: true,
                use_invalid_fee_payer: true,
            }),
        );
    }
}
