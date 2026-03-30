use {
    crate::handshake::{
        AgaveHandshakeError, AgaveTpuToPackSession, AgaveWorkerSession,
        shared::{AgaveSession, MAX_WORKERS},
    },
    agave_scheduler_bindings::PackToWorkerMessage,
    libc::CMSG_LEN,
    nix::sys::socket::{self, ControlMessage, ControlMessageOwned, MsgFlags, UnixAddr},
    rts_alloc::Allocator,
    std::{
        fs::File,
        io::IoSliceMut,
        os::{
            fd::{AsRawFd, FromRawFd},
            unix::net::UnixStream,
        },
    },
};

/// Number of global shared memory objects (allocator + tpu_to_pack + progress_tracker).
const GLOBAL_SHMEM: usize = 3;

/// Maximum control message buffer size assuming [`MAX_WORKERS`].
const CMSG_MAX_SIZE: usize = (GLOBAL_SHMEM + MAX_WORKERS * 2) * 4;

/// Metadata header size: worker_count (u16) + flags (u16).
const METADATA_SIZE: usize = 4;

/// Construct an [`AgaveSession`] from pre-initialized shmem files received via SCM_RIGHTS.
///
/// This is the validator-side counterpart to [`client::setup_session`] — both join
/// already-created shared memory regions rather than creating new ones.
///
/// File order must match [`server::Server::setup_session`] output:
/// - `[0]`: allocator
/// - `[1]`: tpu_to_pack queue
/// - `[2]`: progress_tracker queue
/// - `[3 + 2*i]`: pack_to_worker[i]
/// - `[3 + 2*i + 1]`: worker_to_pack[i]
pub fn agave_session_from_files(
    worker_count: usize,
    flags: u16,
    files: &[File],
) -> Result<AgaveSession, AgaveHandshakeError> {
    let expected = GLOBAL_SHMEM.checked_add(2 * worker_count).unwrap();
    if files.len() != expected {
        return Err(AgaveHandshakeError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "expected {} shmem files, got {}",
                expected,
                files.len()
            ),
        )));
    }

    let allocator_file = &files[0];
    let tpu_to_pack_file = &files[1];
    let progress_tracker_file = &files[2];

    // Join the tpu_to_pack allocator + producer.
    let tpu_to_pack_allocator = Allocator::join(allocator_file)?;
    // SAFETY: The shmem was created by `Server::setup_session` (producer side).
    // We are joining as the producer (validator writes to tpu_to_pack).
    let tpu_to_pack_queue = unsafe { shaq::spsc::Producer::join(tpu_to_pack_file)? };

    // Join the progress_tracker producer.
    // SAFETY: Same — joining producer side of an already-created queue.
    let progress_tracker = unsafe { shaq::spsc::Producer::join(progress_tracker_file)? };

    // Join per-worker sessions.
    let worker_files = &files[GLOBAL_SHMEM..];
    let workers = worker_files
        .chunks(2)
        .map(|window| {
            let [pack_to_worker_file, worker_to_pack_file] = window else {
                panic!();
            };

            let allocator = Allocator::join(allocator_file)?;
            // SAFETY: Joining consumer side of pack_to_worker (validator reads from it).
            let pack_to_worker: shaq::spsc::Consumer<PackToWorkerMessage> =
                unsafe { shaq::spsc::Consumer::join(pack_to_worker_file)? };
            // SAFETY: Joining producer side of worker_to_pack (validator writes to it).
            let worker_to_pack = unsafe { shaq::spsc::Producer::join(worker_to_pack_file)? };

            Ok::<_, AgaveHandshakeError>(AgaveWorkerSession {
                allocator,
                pack_to_worker,
                worker_to_pack,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(AgaveSession {
        flags,
        tpu_to_pack: AgaveTpuToPackSession {
            allocator: tpu_to_pack_allocator,
            producer: tpu_to_pack_queue,
        },
        progress_tracker,
        workers,
    })
}

/// Send shmem file descriptors and metadata over a UDS via `SCM_RIGHTS`.
///
/// Wire format:
/// - `[0..2]`: worker_count (u16 LE)
/// - `[2..4]`: flags (u16 LE)
/// - Attached FDs via SCM_RIGHTS (same order as `Server::setup_session` output)
pub fn send_fds(
    stream: &UnixStream,
    worker_count: u16,
    flags: u16,
    files: &[File],
) -> Result<(), std::io::Error> {
    let mut buf = [0u8; METADATA_SIZE];
    buf[0..2].copy_from_slice(&worker_count.to_le_bytes());
    buf[2..4].copy_from_slice(&flags.to_le_bytes());

    let fds_raw: Vec<_> = files.iter().map(|file| file.as_raw_fd()).collect();
    let iov = [std::io::IoSlice::new(&buf)];
    let cmsgs = [ControlMessage::ScmRights(&fds_raw)];
    let sent =
        socket::sendmsg::<UnixAddr>(stream.as_raw_fd(), &iov, &cmsgs, MsgFlags::empty(), None)
            .map_err(std::io::Error::from)?;
    debug_assert_eq!(sent, METADATA_SIZE);

    Ok(())
}

/// Receive shmem file descriptors and metadata from a UDS via `SCM_RIGHTS`.
///
/// Returns `(worker_count, flags, files)`.
pub fn recv_fds(stream: &UnixStream) -> Result<(u16, u16, Vec<File>), std::io::Error> {
    let mut buf = [0u8; METADATA_SIZE];
    let mut iov = [IoSliceMut::new(&mut buf)];
    // SAFETY: CMSG_LEN is always safe (const expression).
    let mut cmsgs = [0u8; unsafe { CMSG_LEN(CMSG_MAX_SIZE as u32) as usize }];
    let msg = socket::recvmsg::<UnixAddr>(
        stream.as_raw_fd(),
        &mut iov,
        Some(&mut cmsgs),
        MsgFlags::empty(),
    )
    .map_err(std::io::Error::from)?;

    // Parse metadata.
    let data = msg.iovs().next().unwrap();
    if data.len() < METADATA_SIZE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "metadata too short",
        ));
    }
    let worker_count = u16::from_le_bytes([data[0], data[1]]);
    let flags = u16::from_le_bytes([data[2], data[3]]);

    // Extract FDs and wrap in `File` for RAII ownership.
    let mut cmsgs_iter = msg.cmsgs().map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid cmsg")
    })?;
    let fds = match cmsgs_iter.next() {
        Some(ControlMessageOwned::ScmRights(fds)) => fds,
        Some(msg) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unexpected cmsg: {msg:?}"),
            ));
        }
        None => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "no SCM_RIGHTS received",
            ));
        }
    };
    // SAFETY: FDs were just received via ScmRights and are valid.
    let files = fds
        .into_iter()
        .map(|fd| unsafe { File::from_raw_fd(fd) })
        .collect();

    Ok((worker_count, flags, files))
}
