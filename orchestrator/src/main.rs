#![cfg(unix)]

use {
    command_fds::{CommandFdExt, FdMapping},
    std::{
        io::Read,
        os::{
            fd::{AsFd, FromRawFd},
            unix::net::UnixStream,
        },
    },
    tokio::{io::AsyncReadExt, net::UnixStream as TokioUnixStream, process::Command},
};

// TODO: Replace manual parsing with CLAP, just ideally not more clap v2.

fn parse_arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .map(|i| args[i + 1].as_str())
}

fn require_arg<'a>(args: &'a [String], name: &str) -> &'a str {
    parse_arg(args, name).unwrap_or_else(|| panic!("missing {name}"))
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // Parse args.
    let args: Vec<String> = std::env::args().collect();
    let fd: i32 = require_arg(&args, "--orch-fd").parse().expect("bad fd");
    let ipc_path = require_arg(&args, "--ipc-path");
    let scheduler_bin = require_arg(&args, "--external-scheduler-bin");
    let scheduler_config = parse_arg(&args, "--external-scheduler-config");
    eprintln!("[orchestrator] started; orch-fd={fd}");

    // SAFETY: FD was passed to us by the parent process via fork+exec.
    let mut stream = unsafe { UnixStream::from_raw_fd(fd) };

    // Set CLOEXEC so the scheduler child does not inherit this FD.
    nix::fcntl::fcntl(
        &stream,
        nix::fcntl::FcntlArg::F_SETFD(nix::fcntl::FdFlag::FD_CLOEXEC),
    )
    .expect("set CLOEXEC on orchestrator stream");

    // Wait for validator readiness.
    let mut buf = [0u8; 1];
    stream.read_exact(&mut buf).expect("read readiness byte");
    assert_eq!(buf[0], 0x01, "unexpected readiness byte");
    eprintln!("[orchestrator] validator is ready");

    // Spawn the external scheduler (FD is still valid for command-fds to dup).
    let mut scheduler = spawn_scheduler(scheduler_bin, ipc_path, scheduler_config, &stream);
    eprintln!(
        "[orchestrator] spawned scheduler; pid={}",
        scheduler.id().unwrap_or(0)
    );

    // Convert to async for monitoring.
    stream
        .set_nonblocking(true)
        .expect("set nonblocking on orchestrator stream");
    let mut stream = TokioUnixStream::from_std(stream).expect("wrap stream in tokio");

    // Wait for scheduler or Agave to exit.
    tokio::select! {
        status = scheduler.wait() => {
            match status {
                Ok(status) => eprintln!("[orchestrator] scheduler exited; status={status}"),
                Err(err) => eprintln!("[orchestrator] scheduler wait failed; err={err}"),
            }
        }
        () = read_until_eof(&mut stream) => {
            eprintln!("[orchestrator] validator exited (UDS EOF)");
            let _ = scheduler.kill().await;
        }
    }

    eprintln!("[orchestrator] exiting");
}

fn spawn_scheduler(
    bin: &str,
    ipc_path: &str,
    config: Option<&str>,
    orchestrator_stream: &UnixStream,
) -> tokio::process::Child {
    let mut cmd = std::process::Command::new(bin);
    cmd.args(["--bindings-ipc", ipc_path]);
    if let Some(cfg) = config {
        cmd.args(["--config", cfg]);
    }

    // Map the orchestrator UDS to the well-known FD in the child.
    cmd.fd_mappings(vec![FdMapping {
        parent_fd: orchestrator_stream
            .as_fd()
            .try_clone_to_owned()
            .expect("dup fd"),
        child_fd: agave_orchestrator::ORCHESTRATOR_FD,
    }])
    .expect("fd_mappings");

    Command::from(cmd)
        .spawn()
        .unwrap_or_else(|err| panic!("failed to spawn scheduler; bin={bin}; err={err}"))
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
