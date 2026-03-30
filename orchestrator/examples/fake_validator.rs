//! Fake validator for testing the orchestrator.
//! Reads --orch-fd <N>, sends readiness byte, receives shmem FDs,
//! constructs AgaveSession, exercises it briefly, then sleeps.

use agave_scheduling_utils::handshake::orchestrated;
use std::io::Write;
use std::os::fd::FromRawFd;
use std::os::unix::net::UnixStream;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let fd_idx = args.iter().position(|a| a == "--orch-fd").expect("missing --orch-fd");
    let fd: i32 = args[fd_idx + 1].parse().expect("bad fd");

    eprintln!("[validator] started, orch-fd={fd}");

    let mut stream = unsafe { UnixStream::from_raw_fd(fd) };

    // Simulate startup work.
    std::thread::sleep(std::time::Duration::from_secs(1));

    // Signal ready.
    stream.write_all(&[0x01]).expect("write ready");
    eprintln!("[validator] sent ready signal");

    // Receive shmem FDs from orchestrator.
    let (worker_count, flags, files) = orchestrated::recv_fds(&stream).expect("recv shmem fds");
    eprintln!(
        "[validator] received shmem: workers={}, flags={}, fds={}",
        worker_count, flags, files.len(),
    );

    // Construct AgaveSession from the received files.
    let mut session = orchestrated::agave_session_from_files(
        usize::from(worker_count),
        flags,
        &files,
    )
    .expect("agave_session_from_files");
    // Files can be dropped now — mmaps keep the shmem alive.
    drop(files);

    eprintln!(
        "[validator] AgaveSession constructed: {} workers",
        session.workers.len(),
    );

    // Exercise the session: write a progress message.
    let progress = agave_scheduler_bindings::ProgressMessage {
        leader_state: 0,
        current_slot: 42,
        next_leader_slot: u64::MAX,
        leader_range_end: u64::MAX,
        remaining_cost_units: 0,
        current_slot_progress: 0,
    };
    session.progress_tracker.try_write(progress).unwrap();
    session.progress_tracker.commit();
    eprintln!("[validator] wrote ProgressMessage(slot=42)");

    // Run forever (until killed).
    loop {
        std::thread::sleep(std::time::Duration::from_secs(60));
    }
}
