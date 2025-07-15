use {
    crate::attacks::turbine::flood_sigverify::FloodSigverifyArgs, clap::Subcommand, std::fmt::Debug,
};

#[derive(Subcommand, Debug)]
#[command(name = "turbine")]
pub enum TurbineCli {
    /// Runs attack to flood sigverify stage with packets
    #[command(name = "flood-sigverify")]
    FloodSigverify(FloodSigverifyArgs),
}
