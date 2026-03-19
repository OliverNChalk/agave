use {
    crate::TelemetryStamp,
    nix::{
        fcntl::OFlag,
        sys::{
            mman::{shm_open, shm_unlink},
            stat::Mode,
        },
    },
    std::fs::File,
};

const QUEUE_BYTES: usize = 32 * 1024 * 1024;

pub fn try_open(name: &str) -> Option<shaq::Producer<TelemetryStamp>> {
    // Remove previous shm if it exists.
    let _ = shm_unlink(name);

    // Create new shmem as readable writable.
    let fd = shm_open(
        name,
        OFlag::O_CREAT | OFlag::O_RDWR,
        Mode::S_IRUSR | Mode::S_IWUSR | Mode::S_IRGRP | Mode::S_IROTH,
    )
    .expect("shm_open create");
    let file = File::from(fd);

    // Allocate a shaq queue into the file.
    unsafe { shaq::Producer::create(&file, QUEUE_BYTES) }.ok()
}
