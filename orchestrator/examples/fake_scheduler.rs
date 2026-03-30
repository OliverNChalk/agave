//! Fake scheduler for testing the orchestrator.
//! Reads --orch-fd <N>, then exits after 3 seconds to simulate a crash.

use std::os::fd::FromRawFd;
use std::os::unix::net::UnixStream;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let fd_idx = args.iter().position(|a| a == "--orch-fd").expect("missing --orch-fd");
    let fd: i32 = args[fd_idx + 1].parse().expect("bad fd");

    eprintln!("[scheduler] started, orch-fd={fd}");

    // Keep the fd alive so the orchestrator can detect our death via EOF.
    let _stream = unsafe { UnixStream::from_raw_fd(fd) };

    // Simulate a crash after 3 seconds.
    std::thread::sleep(std::time::Duration::from_secs(3));
    eprintln!("[scheduler] crashing!");
    std::process::exit(1);
}
