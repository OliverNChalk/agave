// TODO: This will just straight up break builds for windows in the workspace, maybe its
//       fine if all windows builds do `cargo build -p <package>`?
#![cfg(unix)]

mod args;
mod component;
mod control;

fn main() {
    use {
        crate::control::ControlThread, agave_orchestrator::Config, clap::Parser, clap_v4 as clap,
    };

    let args = args::Args::parse();
    eprintln!("[orchestrator] started; orch-fd={}", args.orch_fd);

    // Load config.
    let config_bytes = std::fs::read(&args.config).expect("failed to read config file");
    let config: Config = toml::from_slice(&config_bytes).expect("failed to parse config");

    ControlThread::run_in_place(&args, config)
}
