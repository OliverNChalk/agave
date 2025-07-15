use {
    super::{
        gossip::args::GossipCli, program_runtime::args::ProgramRuntimeCli,
        turbine::args::TurbineCli,
    },
    clap::Subcommand,
};

#[derive(Subcommand, Debug)]
#[command(name = "attack")]
pub enum AttackCli {
    /// Run an attack targeting gossip
    #[command(subcommand)]
    Gossip(GossipCli),

    /// Run an attack that targets the program runtime.
    #[command(subcommand)]
    ProgramRuntime(ProgramRuntimeCli),

    /// Run an attack that targets the turbine
    #[command(subcommand)]
    Turbine(TurbineCli),
}
