use {
    crate::adversary::repair::{
        DEFAULT_ITERATION_DELAY_US, DEFAULT_PACKETS_PER_PEER_PER_ITERATION,
    },
    clap::{value_t_or_exit, App, AppSettings, Arg, SubCommand},
    solana_adversary::adversary_feature_set::{
        all_enum_variants_as_json_strings, invalidate_leader_block::InvalidationKind,
        repair_packet_flood::FloodStrategy,
    },
    solana_clap_utils::*,
    std::time::Duration,
};

const RPC_ENDPOINT_URL: &str = "http://localhost:8899";

pub fn run_command() -> Result<(), String> {
    let matches = App::new("InvalidatorClient")
        .version("1.0")
        .about("Client for interacting with the Solana Invalidator")
        .subcommand(
            SubCommand::with_name("continuous")
                .about("Continuous cycling through all adversary scenarios")
                .arg(
                    Arg::with_name("runtime")
                        .long("runtime")
                        .default_value("60")
                        .value_name("SECONDS")
                        .validator(|arg| input_validators::is_within_range(arg, 1..))
                        .help("Amount of time in seconds to spend running each scenario"),
                )
                .arg(
                    Arg::with_name("sleeptime")
                        .long("sleeptime")
                        .default_value("5")
                        .value_name("SECONDS")
                        .validator(|arg| input_validators::is_within_range(arg, 1..))
                        .help("Amount of time in seconds to spend sleeping between scenarios"),
                ),
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
                    Arg::with_name("disable")
                        .long("disable")
                        .default_value("false")
                        .value_name("BOOLEAN")
                        .validator(input_validators::is_parsable::<bool>)
                        .help("Whether to disable flooding repair packets"),
                )
                .arg(
                    Arg::with_name("flood_strategy")
                        .long("flood-strategy")
                        .default_value("minimalPackets")
                        .value_name("ENUM STRING")
                        .possible_values(
                            all_enum_variants_as_json_strings::<FloodStrategy>()
                                .iter()
                                .map(|s| s.as_str())
                                .collect::<Vec<&str>>()
                                .as_slice(),
                        )
                        .help("Which strategy to use for flooding repair packets"),
                )
                .arg(
                    Arg::with_name("packets_per_peer_per_iteration")
                        .long("packets-per-peer-per-iteration")
                        .default_value(DEFAULT_PACKETS_PER_PEER_PER_ITERATION)
                        .value_name("NUMBER")
                        .validator(input_validators::is_parsable::<u64>)
                        .help("Number of packets to send to each peer each iteration"),
                )
                .arg(
                    Arg::with_name("iteration_delay_us")
                        .long("iteration-delay-us")
                        .default_value(DEFAULT_ITERATION_DELAY_US)
                        .value_name("MICROSECONDS")
                        .validator(input_validators::is_parsable::<u64>)
                        .help("Delay between iterations in microseconds"),
                )
                .arg(
                    Arg::with_name("target")
                        .long("target")
                        .takes_value(true)
                        .value_name("PUBKEY")
                        .validator(input_validators::is_pubkey)
                        .help("Peer to target with repair packets"),
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
                    Arg::with_name("send_destinations")
                        .long("send-destinations")
                        .takes_value(true)
                        .value_name("SOCKET ADDRESSES")
                        .help("CSV of peer addresses to target with duplicate block"),
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
                        .possible_values(
                            all_enum_variants_as_json_strings::<InvalidationKind>()
                                .iter()
                                .map(|s| s.as_str())
                                .collect::<Vec<&str>>()
                                .as_slice(),
                        )
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
        .setting(AppSettings::SubcommandRequiredElseHelp)
        .get_matches();

    match matches.subcommand() {
        ("continuous", Some(sub_matches)) => {
            let scenario_run_duration =
                Duration::from_secs(value_t_or_exit!(sub_matches, "runtime", u64));
            let rest_between_scenarios_duration =
                Duration::from_secs(value_t_or_exit!(sub_matches, "sleeptime", u64));
            crate::continuous_mode::run_continuous_mode(
                RPC_ENDPOINT_URL,
                scenario_run_duration,
                rest_between_scenarios_duration,
            )
        }
        ("configure-shred-receiver-address", Some(sub_matches)) => {
            crate::adversary::shred_forwarder::configure_shred_receiver_address_args(
                RPC_ENDPOINT_URL,
                sub_matches,
            )
        }
        ("configure-repair-packet-flood", Some(sub_matches)) => {
            crate::adversary::repair::configure_repair_packet_flood_args(
                RPC_ENDPOINT_URL,
                sub_matches,
            )
        }
        ("configure-repair-parameters", Some(sub_matches)) => {
            crate::adversary::repair::configure_repair_parameters_args(
                RPC_ENDPOINT_URL,
                sub_matches,
            )
        }
        ("configure-send-duplicate-blocks", Some(sub_matches)) => {
            crate::adversary::leader_block::configure_send_duplicate_blocks_args(
                RPC_ENDPOINT_URL,
                sub_matches,
            )
        }
        ("configure-invalidate-leader-block", Some(sub_matches)) => {
            crate::adversary::leader_block::configure_invalidate_leader_block_args(
                RPC_ENDPOINT_URL,
                sub_matches,
            )
        }
        ("configure-drop-turbine-votes", Some(sub_matches)) => {
            crate::adversary::drop_turbine_votes::configure_drop_turbine_votes_args(
                RPC_ENDPOINT_URL,
                sub_matches,
            )
        }
        ("configure-packet-drop-parameters", Some(sub_matches)) => {
            crate::adversary::packet_drop::configure_packet_drop_parameters_args(
                RPC_ENDPOINT_URL,
                sub_matches,
            )
        }
        _ => unreachable!(),
    }
}
