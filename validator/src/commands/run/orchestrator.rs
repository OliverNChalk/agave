use {
    command_fds::{CommandFdExt, FdMapping},
    log::info,
    std::{os::unix::net::UnixStream, path::Path, process::Command},
};

/// Spawns the orchestrator and returns agave's side of the UDS pair.
///
/// A background monitoring thread is spawned that will panic (triggering the
/// panic hook for telemetry) if the orchestrator exits unexpectedly.
pub fn spawn_orchestrator(bin: &Path, config_path: &Path) -> UnixStream {
    let (validator_fd, orch_fd) = UnixStream::pair().expect("socketpair failed");

    let mut cmd = Command::new(bin);
    cmd.args(["--config", &config_path.to_string_lossy()]);
    cmd.fd_mappings(vec![FdMapping {
        parent_fd: orch_fd.into(),
        child_fd: agave_orchestrator::ORCHESTRATOR_FD,
    }])
    .expect("fd_mappings failed");

    let mut child = cmd.spawn().unwrap_or_else(|err| {
        panic!(
            "failed to spawn orchestrator; bin={}; err={err}",
            bin.display()
        )
    });

    info!(
        "Spawned orchestrator child process; pid={}; bin={}",
        child.id(),
        bin.display(),
    );

    // Spawn a thread to monitor for unexpected orchestrator exit.
    std::thread::Builder::new()
        .name("orchMonitor".to_string())
        .spawn(move || match child.wait() {
            Ok(status) => panic!("orchestrator exited unexpectedly; status={status}"),
            Err(err) => panic!("orchestrator waitpid failed; err={err}"),
        })
        .expect("failed to spawn orchestrator monitor thread");

    validator_fd
}
