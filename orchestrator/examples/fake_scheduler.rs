//! Fake scheduler for testing the orchestrator.
//! Reads --orch-fd <N>, receives shmem FDs, constructs ClientSession,
//! reads from queues, then exits after 3 seconds to simulate a crash.

use agave_scheduling_utils::handshake::{ClientLogon, client, orchestrated};
use std::os::fd::FromRawFd;
use std::os::unix::net::UnixStream;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let fd_idx = args.iter().position(|a| a == "--orch-fd").expect("missing --orch-fd");
    let fd: i32 = args[fd_idx + 1].parse().expect("bad fd");

    eprintln!("[scheduler] started, orch-fd={fd}");

    // Keep the fd alive so the orchestrator can detect our death via EOF.
    let stream = unsafe { UnixStream::from_raw_fd(fd) };

    // Receive shmem FDs from orchestrator.
    let (worker_count, flags, files) = orchestrated::recv_fds(&stream).expect("recv shmem fds");
    eprintln!(
        "[scheduler] received shmem: workers={}, flags={}, fds={}",
        worker_count, flags, files.len(),
    );

    // Construct ClientSession using the same logon parameters.
    // The scheduler needs to know the capacities used by the orchestrator;
    // for this demo we use defaults that match the orchestrator's defaults.
    let logon = ClientLogon {
        worker_count: usize::from(worker_count),
        allocator_size: 0, // not used by client::setup_session
        allocator_handles: 1,
        tpu_to_pack_capacity: 0, // not used by client::setup_session
        progress_tracker_capacity: 0, // not used by client::setup_session
        pack_to_worker_capacity: 0, // not used by client::setup_session
        worker_to_pack_capacity: 0, // not used by client::setup_session
        flags,
    };
    let mut session = client::setup_session(&logon, files).expect("client::setup_session");
    eprintln!(
        "[scheduler] ClientSession constructed: {} workers, {} allocators",
        session.workers.len(),
        session.allocators.len(),
    );

    // Try to read progress messages from the validator.
    session.progress_tracker.sync();
    if let Some(msg) = session.progress_tracker.try_read() {
        eprintln!("[scheduler] got ProgressMessage: slot={}", msg.current_slot);
        session.progress_tracker.finalize();
    } else {
        eprintln!("[scheduler] no progress messages yet");
    }

    // Simulate a crash after 3 seconds.
    std::thread::sleep(std::time::Duration::from_secs(3));
    eprintln!("[scheduler] crashing!");
    std::process::exit(1);
}
