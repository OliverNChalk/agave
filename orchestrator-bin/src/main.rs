// TODO: This will just straight up break builds for windows in the workspace, maybe its
//       fine if all windows builds do `cargo build -p <package>`?
#![cfg(unix)]

mod config;

use {
    crate::config::Config,
    agave_orchestrator::SessionHeader,
    clap::Parser,
    clap_v4 as clap,
    command_fds::{CommandFdExt, FdMapping},
    std::{
        io::Read,
        os::{fd::FromRawFd, unix::net::UnixStream},
        path::PathBuf,
    },
    tokio::{io::AsyncReadExt, net::UnixStream as TokioUnixStream, process::Command},
};

#[derive(Parser)]
#[command(name = "agave-orchestrator")]
struct Args {
    /// File descriptor for the validator<>orchestrator UDS.
    #[arg(long = "orch-fd")]
    orch_fd: i32,

    /// Path to the orchestrator TOML config file.
    #[arg(long)]
    config: PathBuf,
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let args = Args::parse();
    eprintln!("[orchestrator] started; orch-fd={}", args.orch_fd);

    // Load config.
    let config_bytes = std::fs::read(&args.config).expect("failed to read config file");
    let config: Config = toml::from_slice(&config_bytes).expect("failed to parse config");

    // SAFETY: FD was passed to us by the parent process via fork+exec.
    let mut validator_rx = unsafe { UnixStream::from_raw_fd(args.orch_fd) };

    // Set CLOEXEC so the scheduler child does not inherit this FD.
    nix::fcntl::fcntl(
        &validator_rx,
        nix::fcntl::FcntlArg::F_SETFD(nix::fcntl::FdFlag::FD_CLOEXEC),
    )
    .expect("set CLOEXEC on validator stream");

    // Grab scheduler topology.
    let topology = &config.topology.scheduler;
    let header = SessionHeader::new(
        topology.worker_count,
        topology.allocator_handles,
        topology.flags,
    );

    // Allocate all shared memory regions.
    let files = agave_orchestrator::create_session(topology);
    eprintln!("[orchestrator] created shmem; fds={}", files.len(),);

    // Send shmem FDs to agave.
    agave_orchestrator::send_session(&validator_rx, &files, header);
    eprintln!("[orchestrator] sent session to validator");

    // Wait for agave readiness (banking stage has switched to external mode).
    let mut buf = [0u8; 1];
    validator_rx
        .read_exact(&mut buf)
        .expect("read readiness byte");
    assert_eq!(buf[0], 0x01, "unexpected readiness byte");
    eprintln!("[orchestrator] validator is ready");

    // Create a fresh UDS pair for orchestrator↔scheduler communication.
    let (scheduler_rx, scheduler_tx) =
        UnixStream::pair().expect("create orchestrator↔scheduler UDS pair");

    // Spawn the external scheduler, passing the scheduler end at the well-known fd.
    let scheduler_pid = spawn_scheduler(&config, scheduler_tx);
    eprintln!("[orchestrator] spawned scheduler; pid={scheduler_pid}");

    // Send shmem FDs to scheduler.
    agave_orchestrator::send_session(&scheduler_rx, &files, header);
    eprintln!("[orchestrator] sent session to scheduler");

    // Convert both streams to async for monitoring.
    validator_rx.set_nonblocking(true).unwrap();
    scheduler_rx.set_nonblocking(true).unwrap();
    let mut validator_rx = TokioUnixStream::from_std(validator_rx).unwrap();
    let mut scheduler_rx = TokioUnixStream::from_std(scheduler_rx).unwrap();

    // Monitor both: scheduler UDS EOF and validator UDS EOF.
    tokio::select! {
        () = read_until_eof(&mut scheduler_rx) => {
            eprintln!("[orchestrator] scheduler exited (UDS EOF)");
            read_until_eof(&mut validator_rx).await;
            eprintln!("[orchestrator] validator exited (UDS EOF)");
        }
        () = read_until_eof(&mut validator_rx) => {
            eprintln!("[orchestrator] validator exited (UDS EOF)");
            read_until_eof(&mut scheduler_rx).await;
            eprintln!("[orchestrator] scheduler exited (UDS EOF)");
        }
    }

    eprintln!("[orchestrator] exiting");
}

fn spawn_scheduler(config: &Config, scheduler_tx: UnixStream) -> u32 {
    let bin = &config.scheduler.bin;
    let mut cmd = std::process::Command::new(bin);
    if let Some(cfg) = &config.scheduler.config {
        cmd.args(["--config", &cfg.to_string_lossy()]);
    }

    // Map the scheduler end of the UDS pair to the well-known fd in the child.
    cmd.fd_mappings(vec![FdMapping {
        parent_fd: scheduler_tx.into(),
        child_fd: agave_orchestrator::ORCHESTRATOR_FD,
    }])
    .unwrap();

    let child = Command::from(cmd).spawn().unwrap_or_else(|err| {
        panic!(
            "failed to spawn scheduler; bin={}; err={err}",
            bin.display()
        )
    });

    // SAFETY: We just spawned and haven't polled to completion, so id() is always Some.
    // It's None only after the process has been reaped (to prevent PID reuse bugs).
    child.id().expect("we haven't polled to completion")
}

async fn read_until_eof(stream: &mut TokioUnixStream) {
    let mut buf = [0u8; 64];
    loop {
        match stream.read(&mut buf).await {
            Ok(0) => return,
            Ok(_) => continue,
            Err(err) => {
                eprintln!("[orchestrator] UDS read error; err={err}");
                return;
            }
        }
    }
}
