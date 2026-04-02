#![cfg(unix)]

mod config;

use {
    crate::config::Config,
    command_fds::{CommandFdExt, FdMapping},
    std::{
        io::Read,
        os::{fd::FromRawFd, unix::net::UnixStream},
    },
    tokio::{io::AsyncReadExt, net::UnixStream as TokioUnixStream, process::Command},
};

// TODO: Replace manual parsing with CLAP, just ideally not more clap v2.

fn parse_arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).map(|i| {
        args.get(i + 1)
            .unwrap_or_else(|| panic!("{name} value missing"))
            .as_str()
    })
}

fn require_arg<'a>(args: &'a [String], name: &str) -> &'a str {
    parse_arg(args, name).unwrap_or_else(|| panic!("missing {name}"))
}

fn main() {
    // Parse args.
    let args: Vec<String> = std::env::args().collect();
    let fd: i32 = require_arg(&args, "--orch-fd").parse().expect("bad fd");
    let ipc_path = require_arg(&args, "--ipc-path");
    let scheduler_bin = require_arg(&args, "--external-scheduler-bin");
    let scheduler_config = parse_arg(&args, "--external-scheduler-config");
    eprintln!("[orchestrator] started; orch-fd={fd}");

    // Load config.
    let config = std::fs::read(&args.config).unwrap();
    let config: Config = serde_yaml::from_slice(&config).unwrap();

    // SAFETY: FD was passed to us by the parent process via fork+exec.
    let mut validator_rx = unsafe { UnixStream::from_raw_fd(fd) };

    // Set CLOEXEC so the scheduler child does not inherit this FD.
    nix::fcntl::fcntl(
        &validator_rx,
        nix::fcntl::FcntlArg::F_SETFD(nix::fcntl::FdFlag::FD_CLOEXEC),
    )
    .expect("set CLOEXEC on validator stream");

    // Wait for validator readiness.
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
    let scheduler_pid = spawn_scheduler(scheduler_bin, ipc_path, scheduler_config, scheduler_tx);
    eprintln!("[orchestrator] spawned scheduler; pid={scheduler_pid}");

    // Convert both streams to async for monitoring.
    validator_rx.set_nonblocking(true).unwrap();
    scheduler_rx.set_nonblocking(true).unwrap();
    let mut validator_rx = TokioUnixStream::from_std(validator_rx).unwrap();
    let mut scheduler_rx = TokioUnixStream::from_std(scheduler_rx).unwrap();

    // Monitor both: scheduler UDS EOF and validator UDS EOF.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async move {
        tokio::select! {
            () = read_until_eof(&mut scheduler_rx) => {
                eprintln!("[orchestrator] scheduler exited (UDS EOF)");
            }
            () = read_until_eof(&mut validator_rx) => {
                eprintln!("[orchestrator] validator exited (UDS EOF)");
            }
        }
    });

    eprintln!("[orchestrator] exiting");
}

fn spawn_scheduler(
    bin: &str,
    ipc_path: &str,
    config: Option<&str>,
    scheduler_tx: UnixStream,
) -> u32 {
    let mut cmd = std::process::Command::new(bin);
    cmd.args(["--bindings-ipc", ipc_path]);
    if let Some(cfg) = config {
        cmd.args(["--config", cfg]);
    }

    // Map the scheduler end of the UDS pair to the well-known fd in the child.
    cmd.fd_mappings(vec![FdMapping {
        parent_fd: scheduler_tx.into(),
        child_fd: agave_orchestrator::ORCHESTRATOR_FD,
    }])
    .unwrap();

    let child = Command::from(cmd)
        .spawn()
        .unwrap_or_else(|err| panic!("failed to spawn scheduler; bin={bin}; err={err}"));

    child.id().unwrap()
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
