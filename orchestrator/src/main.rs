#![cfg(unix)]

use std::{
    io::Read,
    os::{fd::FromRawFd, unix::net::UnixStream},
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

fn main() {
    // Parse args.
    let args: Vec<String> = std::env::args().collect();
    let fd: i32 = require_arg(&args, "--orch-fd").parse().expect("bad fd");
    let ipc_path = require_arg(&args, "--ipc-path");
    let scheduler_bin = require_arg(&args, "--external-scheduler-bin");
    let scheduler_config = parse_arg(&args, "--external-scheduler-config");
    eprintln!("[orchestrator] started; orch-fd={fd}");

    // SAFETY: fd was passed to us by the parent process via fork+exec.
    let mut stream = unsafe { UnixStream::from_raw_fd(fd) };

    // Set CLOEXEC so the scheduler child does not inherit this fd.
    nix::fcntl::fcntl(&stream, nix::fcntl::FcntlArg::F_SETFD(nix::fcntl::FdFlag::FD_CLOEXEC))
        .expect("set CLOEXEC on orchestrator stream");

    // Wait for validator readiness.
    let mut buf = [0u8; 1];
    stream.read_exact(&mut buf).expect("read readiness byte");
    assert_eq!(buf[0], 0x01, "unexpected readiness byte");
    eprintln!("[orchestrator] validator is ready");

    // Spawn the external scheduler.
    //
    // SAFETY: Orchestrator is single threaded.
    let scheduler_pid = unsafe { spawn_scheduler(scheduler_bin, ipc_path, scheduler_config) };
    eprintln!("[orchestrator] spawned scheduler; pid={scheduler_pid}");

    // Monitor the scheduler.
    match nix::sys::wait::waitpid(scheduler_pid, None) {
        Ok(status) => {
            eprintln!("[orchestrator] scheduler exited; status={status:?}");
        }
        Err(err) => {
            eprintln!("[orchestrator] waitpid failed; err={err}");
        }
    }

    // Wait for validator exit (EOF on UDS).
    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(_) => continue,
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(err) => {
                eprintln!("[orchestrator] UDS read error; err={err}");
                break;
            }
        }
    }
    eprintln!("[orchestrator] validator exited");

    eprintln!("[orchestrator] exiting");
}

unsafe fn spawn_scheduler(bin: &str, ipc_path: &str, config: Option<&str>) -> nix::unistd::Pid {
    use nix::unistd::{ForkResult, execv, fork};

    let c_bin = std::ffi::CString::new(bin).unwrap();
    let mut argv = vec![
        c_bin.clone(),
        std::ffi::CString::new("--bindings-ipc").unwrap(),
        std::ffi::CString::new(ipc_path).unwrap(),
    ];
    if let Some(cfg) = config {
        argv.push(std::ffi::CString::new("--config").unwrap());
        argv.push(std::ffi::CString::new(cfg).unwrap());
    }

    // SAFETY: The orchestrator is single-threaded at this point.
    match unsafe { fork() }.expect("fork failed") {
        ForkResult::Child => {
            let err = execv(&c_bin, &argv).err().unwrap();
            panic!("execv failed; bin={bin}; err={err:?}");
        }
        ForkResult::Parent { child } => child,
    }
}
