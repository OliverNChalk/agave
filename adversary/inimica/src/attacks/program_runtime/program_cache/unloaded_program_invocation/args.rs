use {
    crate::{args::AccountsFileArgs, programs::KnownPrograms},
    clap_v4::{self as clap, Args, ValueEnum},
    humantime::Duration,
    std::time::Duration as StdDuration,
    strum_macros::Display,
};

#[derive(Args, Debug)]
pub struct UnloadedProgramInvocationArgs {
    #[command(flatten)]
    pub accounts: AccountsFileArgs,

    /// A loader to use in the attack.
    #[arg(long, default_value_t = Loader::default())]
    pub loader: Loader,

    /// A program to use in the attack.
    #[arg(long, default_value_t = Program::default())]
    pub program: Program,

    /// Time duration for the whole attack.
    #[arg(long, value_name = "TIME", default_value_t = StdDuration::from_secs(10 * 60).into())]
    pub total_duration: Duration,

    /// Number of slots to wait between the program deployment and execution.
    ///
    /// Should be long enough for there to be a high probability of the program to be unloaded
    /// before it is executed.
    #[arg(long, value_name = "SLOTS", default_value_t = 10)]
    pub execution_delay: u64,

    /// Maximum number of parallel program deployments that should be running concurrently.
    #[arg(long, value_name = "COUNT", default_value_t = 100)]
    pub parallel_deployments: usize,

    /// Maximum number of programs that can be deployed during a single attack.
    ///
    /// As programs are deployed continuously, but are invoked without any back pressure, except for
    /// the invocation network call delay, the number of deployed programs may grow without any
    /// reasonable constraints.  And if too many programs are pending to be called, attacker
    /// bandwidth could be exceeded, causing chaotic behavior.
    #[arg(long, value_name = "COUNT", default_value_t = 1000)]
    pub max_deployed_programs: usize,

    /// Minimum duration of a single attack program call loop.
    ///
    /// The attack is invoking deployed programs in a loop.  Each loop iteration lasting at least
    /// this long and invoking no more than `max_calls_per_iteration`.
    #[arg(long, value_name = "TIME", default_value_t = StdDuration::from_millis(100).into())]
    pub min_call_iteration_duration: Duration,

    /// Maximum number of programs to call in a single attack loop.
    ///
    /// All programs that are ready to be invoked are invoked in a loop, one by one.  In order to
    /// avoid queuing too many calls, this parameter, together with `min_call_iteration_duration`
    /// specify the maximum rate of call transactions.
    ///
    /// The default of 1000 allows the attack to call at a speed of up to 10k TPS.
    #[arg(long, value_name = "COUNT", default_value_t = 1000)]
    pub max_calls_per_iteration: usize,

    /// When `total_duration` ends, the attacker will delete all the deployed programs, reclaiming
    /// most of the SOL that was used for rent payments for the program data storage.
    ///
    /// For loader v3 programs, a small amount of SOL per program will still be left uncovered, as
    /// we only delete the account holding program bytecode, but not the account that represents
    /// the program state itself.
    ///
    /// For certain environments, this cleanup can be unnecessary.
    #[arg(long, default_value_t = false)]
    pub skip_program_cleanup: bool,
}

#[derive(Default, Debug, Display, Clone, Copy, PartialEq, Eq, ValueEnum)]
/// A program loader to use in the attack.
#[strum(serialize_all = "kebab-case")]
pub enum Loader {
    /// Use loader v2.  Know to be vulnerable.  It has being disabled, in v1.17.20 and on the master
    /// branch.
    V2,

    /// Use an upgradable loader, aka loader v3.  This loader should not be vulnerable.
    #[default]
    V3,
}

#[derive(Default, Debug, Display, Clone, Copy, PartialEq, Eq, ValueEnum)]
/// A program to use in the attack.
#[strum(serialize_all = "kebab-case")]
pub enum Program {
    /// The smallest program.  As the attack does not need for the program to perform any action,
    /// this is the optimal target.
    #[default]
    Noop,
}

impl From<Program> for KnownPrograms {
    fn from(value: Program) -> Self {
        match value {
            Program::Noop => KnownPrograms::Noop,
        }
    }
}
