use {
    super::program_cache::args::ProgramCacheCli,
    clap_v4::{self as clap, Subcommand},
};

#[derive(Subcommand, Debug)]
#[command(name = "program-runtime")]
pub enum ProgramRuntimeCli {
    #[command(subcommand)]
    /// Runs attacks related to the program cache.
    ProgramCache(ProgramCacheCli),
}
