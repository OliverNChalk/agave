use {
    super::{flood_shreds::args::FloodShredsCli, program_runtime::args::ProgramRuntimeCli},
    clap::Subcommand,
};

#[derive(Subcommand, Debug)]
#[command(name = "attack")]
pub enum AttackCli {
    /// Run an attack that targets the program runtime.
    #[command(subcommand)]
    ProgramRuntime(ProgramRuntimeCli),

    /// Run an attack that floods a validator with shreds
    #[command(subcommand)]
    FloodShreds(FloodShredsCli),
}
