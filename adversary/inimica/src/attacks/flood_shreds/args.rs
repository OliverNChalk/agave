use {super::invalid_shreds::InvalidShredsArgs, clap::Subcommand};

#[derive(Subcommand, Debug)]
#[command(name = "flood-shreds")]
pub enum FloodShredsCli {
    #[command(name = "invalid-shreds")]
    /// Runs attacks related to flooding invalid shreds
    InvalidShreds(InvalidShredsArgs),
}
