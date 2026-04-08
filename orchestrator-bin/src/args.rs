use {clap::Parser, clap_v4 as clap, std::path::PathBuf};

#[derive(Parser)]
#[command(name = "agave-orchestrator")]
pub(crate) struct Args {
    /// File descriptor for the validator<>orchestrator UDS.
    #[arg(long = "orch-fd")]
    pub(crate) orch_fd: i32,

    /// Path to the orchestrator TOML config file.
    #[arg(long)]
    pub(crate) config: PathBuf,
}
