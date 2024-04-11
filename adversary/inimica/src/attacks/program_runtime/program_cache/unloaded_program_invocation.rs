//! Attack that calls a program after it is unloaded.
//!
//! For details, see [`docs/invalidator/inimica.md`].

use {
    crate::{args::AccountsFileArgs, programs::program_elf, report_attack_execution},
    log::info,
    solana_adversary::accounts_file::AccountsFile,
    solana_commitment_config::CommitmentConfig,
    solana_metrics::metrics::MetricsSender,
    solana_rpc_client::nonblocking::rpc_client::RpcClient,
};

pub mod args;
mod loader_v3;

use {
    args::{Loader, UnloadedProgramInvocationArgs},
    loader_v3::{attack as attack_v3, Config as AttackV3Config},
};

pub async fn run(
    metrics: &impl MetricsSender,
    json_rpc_url: &str,
    args: UnloadedProgramInvocationArgs,
) -> Result<(), String> {
    let UnloadedProgramInvocationArgs {
        accounts: AccountsFileArgs { accounts },
        loader,
        program,
        total_duration,
        execution_delay,
        parallel_deployments,
        max_deployed_programs,
        min_call_iteration_duration,
        max_calls_per_iteration,
        skip_program_cleanup,
    } = args;

    let AccountsFile { payers, .. } = accounts;
    let elf = program_elf(program.into());

    info!(
        "JSON RPC url: {json_rpc_url}\nGoing to attack using {program:?}, for \
         {total_duration}.\nWith the execution delay of {execution_delay} blocks.\nProgram ELF \
         length: {}\nGot {} payers.",
        elf.len(),
        payers.len(),
    );

    let rpc_client =
        RpcClient::new_with_commitment(json_rpc_url.to_owned(), CommitmentConfig::confirmed());

    match loader {
        Loader::V2 => panic!("Attack on the loader v2 is not implemented yet"),
        Loader::V3 => {
            report_attack_execution(
                metrics,
                "program_runtime.program_cache.unloaded_program_invocation.loader_v3",
                attack_v3(
                    AttackV3Config {
                        payers: &payers,
                        program: &elf,
                        total_duration: total_duration.into(),
                        execution_delay,
                        parallel_deployments,
                        max_deployed_programs,
                        min_call_iteration_duration: min_call_iteration_duration.into(),
                        max_calls_per_iteration,
                        skip_program_cleanup,
                    },
                    metrics,
                    &rpc_client,
                ),
            )
            .await
        }
    }
}
