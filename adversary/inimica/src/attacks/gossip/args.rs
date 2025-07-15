use {
    crate::attacks::gossip::flood_sigverify::FloodSigverifyArgs, clap::Subcommand, std::fmt::Debug,
};

#[derive(Subcommand, Debug)]
#[command(name = "gossip")]
pub enum GossipCli {
    /// Runs attack to flood sigverify stage with packets
    #[command(name = "flood-sigverify")]
    FloodSigverify(FloodSigverifyArgs),
}
