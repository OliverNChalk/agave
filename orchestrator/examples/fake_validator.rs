//! Fake validator for testing the orchestrator.
//! Reads --orch-fd <N>, sends readiness byte, then sleeps.

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

    // Run forever (until killed).
    loop {
        std::thread::sleep(std::time::Duration::from_secs(60));
    }
}
