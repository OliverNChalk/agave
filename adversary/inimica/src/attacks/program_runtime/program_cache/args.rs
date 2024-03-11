use {super::unloaded_program_invocation::args::UnloadedProgramInvocationArgs, clap::Subcommand};

#[derive(Subcommand, Debug)]
#[command(name = "program-cache")]
pub enum ProgramCacheCli {
    #[command(name = "unloaded-program-invocation")]
    /// Run the "Invocation of an unloaded program" attack.
    UnloadedProgramInvocation(UnloadedProgramInvocationArgs),
}
