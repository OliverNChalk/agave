//! Minimal orchestrator demo.
//!
//! Spawns agave-validator with an inherited UDS fd, waits for a readiness
//! signal over that UDS, then spawns the configured external scheduler
//! (also with its own UDS fd).  Allocates shared memory via
//! `Server::setup_session` and passes the FDs to both children via
//! SCM_RIGHTS.  Monitors both — if either exits the orchestrator tears
//! everything down.

use agave_scheduling_utils::handshake::{ClientLogon, orchestrated, server::Server};
use nix::fcntl::{fcntl, FcntlArg, FdFlag};
use nix::sys::socket::{self, AddressFamily, SockFlag, SockType};
use nix::sys::wait::{self, WaitStatus};
use nix::unistd::{self, ForkResult, Pid};
use std::io::{self, Read};
use std::os::fd::{BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;
use std::process;

// ── Config ──────────────────────────────────────────────────────────────

struct Config {
    validator_bin: String,
    validator_args: Vec<String>,
    scheduler_bin: String,
    scheduler_args: Vec<String>,

    // Shmem parameters.
    worker_count: usize,
    allocator_size: usize,
    allocator_handles: usize,
    tpu_to_pack_capacity: usize,
    progress_tracker_capacity: usize,
    pack_to_worker_capacity: usize,
    worker_to_pack_capacity: usize,
    flags: u16,
}

fn parse_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|v| v.parse().unwrap_or_else(|_| panic!("{name} must be a valid usize")))
        .unwrap_or(default)
}

fn config_from_env() -> Config {
    Config {
        validator_bin: std::env::var("VALIDATOR_BIN")
            .unwrap_or_else(|_| "agave-validator".into()),
        validator_args: std::env::var("VALIDATOR_ARGS")
            .map(|s| s.split_whitespace().map(String::from).collect())
            .unwrap_or_default(),
        scheduler_bin: std::env::var("SCHEDULER_BIN")
            .unwrap_or_else(|_| "fd-scheduler".into()),
        scheduler_args: std::env::var("SCHEDULER_ARGS")
            .map(|s| s.split_whitespace().map(String::from).collect())
            .unwrap_or_default(),

        worker_count: parse_env_usize("WORKER_COUNT", 4),
        allocator_size: parse_env_usize("ALLOCATOR_SIZE", 1024 * 1024 * 1024),
        allocator_handles: parse_env_usize("ALLOCATOR_HANDLES", 1),
        tpu_to_pack_capacity: parse_env_usize("TPU_TO_PACK_CAPACITY", 65536),
        progress_tracker_capacity: parse_env_usize("PROGRESS_TRACKER_CAPACITY", 256),
        pack_to_worker_capacity: parse_env_usize("PACK_TO_WORKER_CAPACITY", 128),
        worker_to_pack_capacity: parse_env_usize("WORKER_TO_PACK_CAPACITY", 256),
        flags: std::env::var("FLAGS")
            .ok()
            .map(|v| v.parse().unwrap_or_else(|_| panic!("FLAGS must be a valid u16")))
            .unwrap_or(0),
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Create a `socketpair(AF_UNIX, SOCK_STREAM, CLOEXEC)` and return both
/// ends as `OwnedFd`.  CLOEXEC is set on *both* — the caller clears it
/// on the child's end right before exec.
fn uds_pair() -> nix::Result<(OwnedFd, OwnedFd)> {
    socket::socketpair(
        AddressFamily::Unix,
        SockType::Stream,
        None,
        SockFlag::SOCK_CLOEXEC,
    )
}

/// Clear CLOEXEC so the fd survives execve.
///
/// # Safety
/// `fd` must be a valid, open file descriptor.
unsafe fn clear_cloexec(fd: RawFd) -> nix::Result<()> {
    fcntl(BorrowedFd::borrow_raw(fd), FcntlArg::F_SETFD(FdFlag::empty()))?;
    Ok(())
}

/// fork + exec a child, passing `child_fd` as an inherited file descriptor.
/// The child receives the fd number via `--orch-fd <N>`.
/// Returns the child PID.  The child side of the fd is consumed.
fn spawn(bin: &str, args: &[String], child_fd: OwnedFd) -> nix::Result<Pid> {
    let raw = child_fd.into_raw_fd(); // keep alive until exec

    // Safety: single-threaded at this point (no async runtime).
    match unsafe { unistd::fork() }? {
        ForkResult::Child => {
            // In child — clear CLOEXEC on the fd we want to pass.
            // Safety: raw is a valid fd we just obtained from into_raw_fd().
            unsafe { clear_cloexec(raw) }.expect("clear_cloexec");

            let fd_str = raw.to_string();
            let c_bin = std::ffi::CString::new(bin).unwrap();

            let mut argv_strings: Vec<String> = Vec::new();
            argv_strings.push(bin.to_string());
            argv_strings.extend_from_slice(args);
            argv_strings.push("--orch-fd".to_string());
            argv_strings.push(fd_str);

            let c_args: Vec<std::ffi::CString> = argv_strings
                .iter()
                .map(|s| std::ffi::CString::new(s.as_str()).unwrap())
                .collect();

            unistd::execvp(&c_bin, &c_args).expect("execvp");
            #[allow(unreachable_code)]
            { process::exit(1); }
        }
        ForkResult::Parent { child } => {
            // Parent — close the child's end of the pair.
            // Safety: raw is a valid fd we own in the parent.
            unsafe { nix::libc::close(raw) };
            Ok(child)
        }
    }
}

// ── Readiness handshake ────────────────────────────────────────────────

/// Block until the validator sends a one-byte "ready" message (0x01) on
/// our end of the UDS.  This is the point where the validator indicates
/// it can accept external scheduler connections.
fn wait_for_ready(orch_end: &mut UnixStream) -> io::Result<()> {
    let mut buf = [0u8; 1];
    orch_end.read_exact(&mut buf)?;
    if buf[0] != 0x01 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unexpected ready byte: 0x{:02x}", buf[0]),
        ));
    }
    eprintln!("[orch] validator signalled ready");
    Ok(())
}

// ── Shmem ──────────────────────────────────────────────────────────────

fn create_logon(cfg: &Config) -> ClientLogon {
    ClientLogon {
        worker_count: cfg.worker_count,
        allocator_size: cfg.allocator_size,
        allocator_handles: cfg.allocator_handles,
        tpu_to_pack_capacity: cfg.tpu_to_pack_capacity,
        progress_tracker_capacity: cfg.progress_tracker_capacity,
        pack_to_worker_capacity: cfg.pack_to_worker_capacity,
        worker_to_pack_capacity: cfg.worker_to_pack_capacity,
        flags: cfg.flags,
    }
}

/// Send shmem FDs to a child over its orch UDS connection.
fn send_shmem_fds(
    stream: &UnixStream,
    cfg: &Config,
    files: &[std::fs::File],
) -> io::Result<()> {
    let worker_count = u16::try_from(cfg.worker_count)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "worker_count exceeds u16"))?;
    orchestrated::send_fds(stream, worker_count, cfg.flags, files)
}

// ── Process monitoring ─────────────────────────────────────────────────

/// Wait for either child to exit.
fn wait_any() -> nix::Result<WaitStatus> {
    wait::waitpid(None, None)
}

fn shutdown(pids: &[Pid]) {
    for &pid in pids {
        let _ = nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGTERM);
    }
    // Reap stragglers.
    for &pid in pids {
        let _ = wait::waitpid(pid, None);
    }
}

// ── Main ───────────────────────────────────────────────────────────────

fn main() {
    let cfg = config_from_env();

    // 1. Create UDS pairs.
    let (orch_val, child_val) = uds_pair().expect("socketpair");
    let (orch_sched, child_sched) = uds_pair().expect("socketpair");

    // 2. Spawn validator with its UDS fd.
    eprintln!("[orch] spawning validator: {}", cfg.validator_bin);
    let val_pid = spawn(&cfg.validator_bin, &cfg.validator_args, child_val)
        .expect("spawn validator");
    eprintln!("[orch] validator pid={val_pid}");

    // 3. Wait for validator to signal readiness.
    //    Safety: orch_val is a valid connected socket we own.
    let mut val_stream = unsafe { UnixStream::from_raw_fd(orch_val.into_raw_fd()) };
    wait_for_ready(&mut val_stream).expect("validator readiness");

    // 4. Create shmem via Server::setup_session.
    let logon = create_logon(&cfg);
    eprintln!(
        "[orch] creating shmem: workers={}, allocator_size={}",
        logon.worker_count, logon.allocator_size,
    );
    let (_session, shmem_files) = Server::setup_session(logon).expect("setup_session");
    // Drop the AgaveSession — mmaps are unmapped but the memfd data persists
    // because the File handles keep the FDs alive.
    drop(_session);
    eprintln!("[orch] shmem created: {} fds", shmem_files.len());

    // 5. Send shmem FDs to validator via SCM_RIGHTS.
    send_shmem_fds(&val_stream, &cfg, &shmem_files).expect("send shmem to validator");
    eprintln!("[orch] sent shmem fds to validator");

    // 6. Spawn scheduler with its UDS fd.
    eprintln!("[orch] spawning scheduler: {}", cfg.scheduler_bin);
    let sched_pid = spawn(&cfg.scheduler_bin, &cfg.scheduler_args, child_sched)
        .expect("spawn scheduler");
    eprintln!("[orch] scheduler pid={sched_pid}");

    // 7. Send shmem FDs to scheduler via SCM_RIGHTS.
    let sched_stream = unsafe { UnixStream::from_raw_fd(orch_sched.into_raw_fd()) };
    send_shmem_fds(&sched_stream, &cfg, &shmem_files).expect("send shmem to scheduler");
    eprintln!("[orch] sent shmem fds to scheduler");

    // 8. Drop shmem files — children have their copies via SCM_RIGHTS.
    drop(shmem_files);

    // Keep the orchestrator's UDS ends alive for EOF detection.
    let _val_stream = val_stream;
    let _sched_stream = sched_stream;

    // 9. Monitor — if either exits, tear down the other.
    eprintln!("[orch] monitoring children");
    let all = [val_pid, sched_pid];
    match wait_any() {
        Ok(status) => {
            eprintln!("[orch] child exited: {status:?}");
            eprintln!("[orch] shutting down remaining children");
            shutdown(&all);
        }
        Err(e) => {
            eprintln!("[orch] wait error: {e}");
            shutdown(&all);
        }
    }

    process::exit(1);
}
