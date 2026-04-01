use std::{
    io::Read,
    os::{fd::FromRawFd, unix::net::UnixStream},
};

fn main() {
    // TODO: Can we use clap-v4, its so much nicer => probably not cause of workspace :(.
    let args: Vec<String> = std::env::args().collect();
    let fd_idx = args
        .iter()
        .position(|a| a == "--orch-fd")
        .expect("missing --orch-fd");
    let fd: i32 = args[fd_idx + 1].parse().expect("bad fd");

    eprintln!("[orchestrator] started; orch-fd={fd}");

    // SAFETY: fd was passed to us by the parent process via fork+exec.
    let mut stream = unsafe { UnixStream::from_raw_fd(fd) };

    // Wait for validator readiness.
    let mut buf = [0u8; 1];
    stream.read_exact(&mut buf).expect("read readiness byte");
    assert_eq!(buf[0], 0x01, "unexpected readiness byte");
    eprintln!("[orchestrator] validator is ready");

    // Wait for validator exit.
    loop {
        match stream.read(&mut buf) {
            Ok(len) => {
                assert_eq!(len, 0);

                break;
            }
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(err) => panic!("Unexpected IO err; err={err}"),
        }
    }

    // Log & exit.
    eprintln!("[orchestrator] validator exited");
}
