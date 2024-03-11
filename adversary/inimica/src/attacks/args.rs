use {super::program_runtime::args::ProgramRuntimeCli, clap::Subcommand};

#[derive(Subcommand, Debug)]
#[command(name = "attack")]
pub enum AttackCli {
    #[command(subcommand)]
    /// Run an attack that targets the program runtime.
    ProgramRuntime(ProgramRuntimeCli),
}
