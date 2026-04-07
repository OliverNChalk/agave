use {
    log::info,
    nix::{
        fcntl::{FcntlArg, FdFlag, fcntl},
        sys::socket::{self, AddressFamily, SockFlag, SockType},
        unistd::{self, ForkResult},
    },
    std::{
        os::{fd::AsRawFd, unix::net::UnixStream},
        path::Path,
    },
};

/// Spawns the orchestrator and returns agave's side of the UDS pair.
pub fn spawn_orchestrator(bin: &Path, ipc_path: &Path, config_path: &Path) -> UnixStream {
    let (validator_fd, orch_fd) = socket::socketpair(
        AddressFamily::Unix,
        SockType::Stream,
        None,
        SockFlag::SOCK_CLOEXEC,
    )
    .expect("socketpair failed");

    // Construct our execv arguments before forking (allocations are not
    // async-signal-safe and must not happen in the child).
    let bin_str = bin
        .to_str()
        .expect("orchestrator bin path is not valid UTF-8");
    let ipc_str = ipc_path
        .to_str()
        .expect("ipc path is not valid UTF-8");
    let config_str = config_path
        .to_str()
        .expect("config path is not valid UTF-8");
    let c_bin = std::ffi::CString::new(bin_str).unwrap();
    let args = vec![
        c_bin.clone(),
        std::ffi::CString::new("--orch-fd").unwrap(),
        std::ffi::CString::new(orch_fd.as_raw_fd().to_string()).unwrap(),
        std::ffi::CString::new("--ipc-path").unwrap(),
        std::ffi::CString::new(ipc_str).unwrap(),
        std::ffi::CString::new("--config").unwrap(),
        std::ffi::CString::new(config_str).unwrap(),
    ];

    // SAFETY: Only async-signal-safe operations are performed in the child arm.
    match unsafe { unistd::fork() }.expect("fork failed") {
        ForkResult::Child => {
            // Clear CLOEXEC on orch_fd so it survives execv.
            if fcntl(&orch_fd, FcntlArg::F_SETFD(FdFlag::empty())).is_err() {
                die(b"orchestrator: fcntl CLOEXEC failed\n");
            }

            // Execv into the new binary.
            let _ = unistd::execv(&c_bin, &args);

            // Only reachable if execv fails.
            die(b"orchestrator: execv failed\n");
        }
        ForkResult::Parent { child } => {
            info!(
                "Spawned orchestrator child process; pid={child}; bin={}",
                bin.display()
            );

            UnixStream::from(validator_fd)
        }
    }
}

/// Writes a message to stderr and exits. All operations are async-signal-safe.
fn die(msg: &[u8]) -> ! {
    unsafe {
        libc::write(libc::STDERR_FILENO, msg.as_ptr().cast(), msg.len());
        libc::_exit(1);
    }
}
