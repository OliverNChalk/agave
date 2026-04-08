use {
    crate::{
        args::Args,
        component::{Component, Role},
    },
    agave_orchestrator::{Config, SessionHeader},
    command_fds::{CommandFdExt, FdMapping},
    futures::{StreamExt, stream::FuturesUnordered},
    std::{
        io::Read,
        os::{fd::FromRawFd, unix::net::UnixStream},
    },
    tokio::{net::UnixStream as TokioUnixStream, process::Command, signal::unix::SignalKind},
};

pub(crate) struct ControlThread {
    components: FuturesUnordered<Component>,
}

impl ControlThread {
    pub(crate) fn run_in_place(args: &Args, config: Config) {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let server = rt.block_on(ControlThread::setup(args, config));

        rt.block_on(server.run())
    }

    async fn setup(args: &Args, config: Config) -> Self {
        // Grab scheduler topology.
        let topology = &config.topology.scheduler;
        let header = SessionHeader::new(
            topology.worker_count,
            topology.allocator_handles,
            topology.flags,
        );

        // SAFETY: FD was passed to us by the parent process via fork+exec.
        let mut validator_rx = unsafe { UnixStream::from_raw_fd(args.orch_fd) };

        // Set CLOEXEC so the scheduler child does not inherit this FD.
        nix::fcntl::fcntl(
            &validator_rx,
            nix::fcntl::FcntlArg::F_SETFD(nix::fcntl::FdFlag::FD_CLOEXEC),
        )
        .expect("set CLOEXEC on validator stream");

        // Allocate all shared memory regions.
        let files = agave_orchestrator::create_session(topology);
        eprintln!("[orchestrator] created shmem; fds={}", files.len(),);

        // Send shmem FDs to agave.
        agave_orchestrator::send_session(&validator_rx, &files, header);
        eprintln!("[orchestrator] sent session to validator");

        // Wait for agave readiness (banking stage has switched to external mode).
        let mut buf = [0u8; 1];
        validator_rx
            .read_exact(&mut buf)
            .expect("read readiness byte");
        assert_eq!(buf[0], 0x01, "unexpected readiness byte");
        eprintln!("[orchestrator] validator is ready");

        // Create a fresh UDS pair for orchestrator <> scheduler communication.
        let (scheduler_rx, scheduler_tx) =
            UnixStream::pair().expect("create orchestrator <> scheduler UDS pair");

        // Spawn the external scheduler, passing the scheduler end at the well-known fd.
        let scheduler_pid = Self::spawn_scheduler(&config, scheduler_tx);
        eprintln!("[orchestrator] spawned scheduler; pid={scheduler_pid}");

        // Send shmem FDs to scheduler.
        agave_orchestrator::send_session(&scheduler_rx, &files, header);
        eprintln!("[orchestrator] sent session to scheduler");

        // Convert both streams to async for monitoring.
        validator_rx.set_nonblocking(true).unwrap();
        scheduler_rx.set_nonblocking(true).unwrap();
        let validator_rx = TokioUnixStream::from_std(validator_rx).unwrap();
        let scheduler_rx = TokioUnixStream::from_std(scheduler_rx).unwrap();

        ControlThread {
            components: FuturesUnordered::from_iter([
                Component::new(Role::Validator, validator_rx),
                Component::new(Role::Scheduler, scheduler_rx),
            ]),
        }
    }

    async fn run(mut self) {
        let mut sigterm = tokio::signal::unix::signal(SignalKind::terminate()).unwrap();
        let mut sigint = tokio::signal::unix::signal(SignalKind::interrupt()).unwrap();

        tokio::select! {
            _ = sigterm.recv() => {
                eprintln!("[orchestrator] SIGTERM caught, stopping server");
            },
            _ = sigint.recv() => {
                eprintln!("[orchestrator] SIGINT caught, stopping server");
            },
            opt = self.components.next() => {
                let role = opt.unwrap();
                eprintln!("[orchestrator] Component exited unexpectedly; role={role:?}");
            },
        };

        // Signal shutdown to remaining components.
        for component in self.components.iter_mut() {
            component.shutdown().await;
        }

        // Wait for remaining components to exit.
        while let Some(role) = self.components.next().await {
            eprintln!("[orchestrator] Component exited; role={role:?}");
        }

        eprintln!("[orchestrator] exiting");
    }

    fn spawn_scheduler(config: &Config, scheduler_tx: UnixStream) -> u32 {
        let bin = &config.scheduler.bin;
        let mut cmd = std::process::Command::new(bin);
        if let Some(cfg) = &config.scheduler.config {
            cmd.args(["--config", &cfg.to_string_lossy()]);
        }

        // Map the scheduler end of the UDS pair to the well-known fd in the child.
        cmd.fd_mappings(vec![FdMapping {
            parent_fd: scheduler_tx.into(),
            child_fd: agave_orchestrator::ORCHESTRATOR_FD,
        }])
        .unwrap();

        let child = Command::from(cmd).spawn().unwrap_or_else(|err| {
            panic!(
                "failed to spawn scheduler; bin={}; err={err}",
                bin.display()
            )
        });

        // SAFETY: We just spawned and haven't polled to completion, so id() is always Some.
        // It's None only after the process has been reaped (to prevent PID reuse bugs).
        child.id().expect("we haven't polled to completion")
    }
}
