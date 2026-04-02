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
///
/// # Safety
/// - Must be called before any threads are spawned (allocating CString between fork &
///   execv is not multithread safe).
pub unsafe fn spawn_orchestrator(bin: &Path, extra_args: &[&str]) -> UnixStream {
    let (validator_fd, orch_fd) = socket::socketpair(
        AddressFamily::Unix,
        SockType::Stream,
        None,
        SockFlag::SOCK_CLOEXEC,
    )
    .expect("socketpair failed");

    // let orch_raw = orch_fd.into_raw_fd();

    // SAFETY: Caller ensures no other threads exist.
    match unsafe { unistd::fork() }.expect("fork failed") {
        ForkResult::Child => {
            // Clear CLOEXEC on orch_raw so it survives execv.
            fcntl(&orch_fd, FcntlArg::F_SETFD(FdFlag::empty())).expect("clear CLOEXEC");

            // Execv into the new binary.
            let bin_str = bin
                .to_str()
                .expect("orchestrator bin path is not valid UTF-8");
            let c_bin = std::ffi::CString::new(bin_str).unwrap();
            let mut args = vec![
                c_bin.clone(),
                std::ffi::CString::new("--orch-fd").unwrap(),
                std::ffi::CString::new(orch_fd.as_raw_fd().to_string()).unwrap(),
            ];
            for arg in extra_args {
                args.push(std::ffi::CString::new(*arg).unwrap());
            }
            let err = unistd::execv(&c_bin, &args).err().unwrap();

            // Only reachable if execv fails.
            panic!("execv failed; bin={bin_str}; err={err:?}");
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
