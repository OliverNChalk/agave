use {
    crate::{
        args::Args,
        component::{Component, Role},
    },
    agave_orchestrator::{Config, SessionHeader},
    command_fds::{CommandFdExt, FdMapping},
    futures::{StreamExt, stream::FuturesUnordered},
    nix::unistd::Pid,
    std::{
        io::Read,
        os::{
            fd::{AsFd, FromRawFd},
            unix::net::UnixStream,
        },
        path::Path,
        process::Stdio,
    },
    tokio::{
        io::AsyncReadExt, net::UnixStream as TokioUnixStream, process::Command,
        signal::unix::SignalKind,
    },
};

pub(crate) struct ControlThread {
    args: Args,
    config: Config,

    validator_rx: TokioUnixStream,
    components: FuturesUnordered<Component>,
}

impl ControlThread {
    pub(crate) fn run_in_place(args: Args, config: Config) {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let server = rt.block_on(ControlThread::setup(args, config));

        rt.block_on(server.run())
    }

    async fn setup(args: Args, config: Config) -> Self {
        // SAFETY: FD 3 was mapped by the parent process via command-fds.
        let mut validator_rx =
            unsafe { UnixStream::from_raw_fd(agave_orchestrator::ORCHESTRATOR_FD) };

        // Set CLOEXEC so the scheduler child does not inherit this FD.
        nix::fcntl::fcntl(
            &validator_rx,
            nix::fcntl::FcntlArg::F_SETFD(nix::fcntl::FdFlag::FD_CLOEXEC),
        )
        .expect("set CLOEXEC on validator stream");

        // Wait for agave readiness (banking is ready to start accepting shmem).
        let mut buf = [0u8; 1];
        validator_rx
            .read_exact(&mut buf)
            .expect("read readiness byte");
        assert_eq!(buf[0], 0x01, "unexpected readiness byte");
        log::info!("Validator is ready");

        // Wrap validator stream.
        validator_rx.set_nonblocking(true).unwrap();
        let validator_rx = TokioUnixStream::from_std(validator_rx).unwrap();

        // Setup thread structure.
        let mut thread = ControlThread {
            args,
            config,

            validator_rx,
            components: FuturesUnordered::new(),
        };

        // Spawn initial topology.
        thread.spawn_components();

        // Le go.
        thread
    }

    async fn run(mut self) {
        let mut sigterm = tokio::signal::unix::signal(SignalKind::terminate()).unwrap();
        let mut sigint = tokio::signal::unix::signal(SignalKind::interrupt()).unwrap();
        let mut sighup = tokio::signal::unix::signal(SignalKind::hangup()).unwrap();

        loop {
            tokio::select! {
                _ = sigterm.recv() => {
                    log::info!("SIGTERM caught, stopping");

                    break;
                },
                _ = sigint.recv() => {
                    log::info!("SIGINT caught, stopping");

                    break;
                },
                _ = sighup.recv() => {
                    log::info!("SIGHUP caught, reloading config");
                    if let Err(err) = self.reload(&self.args.config.clone()).await {
                        log::error!("Failed to reload; err={err}");
                    }
                },

                () = Self::read_until_eof(&mut self.validator_rx) => {
                    log::error!("Validator exited unexpectedly");

                    break;
                },
                opt = self.components.next() => {
                    let role = opt.unwrap();
                    log::error!("Component exited unexpectedly; role={role:?}");

                    match self.config.fallback.clone() {
                        Some(fallback) => {
                            log::info!("Falling back; fallback={}", fallback.display());
                            self.reload(&fallback).await.unwrap();
                        },
                        // NB: Shutting down the validator is a massive pain until the
                        // validator becomes our child. So we panic which causes agave
                        // to abort.
                        None => panic!("Can't shutdown our parent"),
                    }
                },
            };
        }

        self.shutdown_components().await;

        log::info!("Exiting");
    }

    async fn reload(&mut self, config: &Path) -> anyhow::Result<()> {
        // Load config.
        let config_bytes = std::fs::read(config)?;
        self.config = toml::from_slice(&config_bytes)?;

        // Tear down existing components.
        self.shutdown_components().await;

        // Spawn new components.
        self.spawn_components();

        Ok(())
    }

    async fn shutdown_components(&mut self) {
        // Signal shutdown to remaining components.
        for component in self.components.iter_mut() {
            component.shutdown();
        }

        // Wait for remaining components to exit.
        while let Some(role) = self.components.next().await {
            log::info!("Component exited; role={role:?}");
        }
    }

    fn spawn_components(&mut self) {
        assert!(self.components.is_empty());

        // Grab scheduler topology.
        let header = SessionHeader::new(
            self.config.topology.scheduler.worker_count,
            self.config.topology.scheduler.allocator_handles,
            self.config.topology.scheduler.flags,
        );

        // Allocate all shared memory regions.
        let files = agave_orchestrator::create_session(&self.config.topology.scheduler);
        log::info!("Created shmem; fds={}", files.len());

        // Send shmem FDs to agave.
        agave_orchestrator::send_session(self.validator_rx.as_fd(), &files, header);
        log::info!("Sent session to validator");

        // Create a fresh UDS pair for orchestrator <> scheduler communication.
        let (scheduler_rx, scheduler_tx) =
            UnixStream::pair().expect("create orchestrator <> scheduler UDS pair");

        // Spawn the external scheduler, passing the scheduler end at the well-known fd.
        let scheduler_pid = Self::spawn_scheduler(&self.config, scheduler_tx);
        log::info!("Spawned scheduler; pid={scheduler_pid}");

        // Send shmem FDs to scheduler.
        agave_orchestrator::send_session(scheduler_rx.as_fd(), &files, header);
        log::info!("Sent session to scheduler");

        // Convert both streams to async for monitoring.
        scheduler_rx.set_nonblocking(true).unwrap();
        let scheduler_rx = TokioUnixStream::from_std(scheduler_rx).unwrap();

        // Store all components for monitoring.
        self.components
            .extend([Component::new(Role::Scheduler, scheduler_pid, scheduler_rx)]);
    }

    fn spawn_scheduler(config: &Config, scheduler_tx: UnixStream) -> Pid {
        let bin = &config.scheduler.bin;
        let mut cmd = std::process::Command::new(bin);

        // Don't inherit stdout/stderr.
        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::null());

        // Set the log path.
        cmd.args(["--logs", &config.scheduler.log.to_string_lossy()]);

        // Set config if provided.
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
        let pid = child.id().expect("we haven't polled to completion");

        Pid::from_raw(pid as i32)
    }

    async fn read_until_eof(stream: &mut TokioUnixStream) {
        let mut buf = [0u8; 64];
        loop {
            match stream.read(&mut buf).await {
                Ok(0) => return,
                Ok(_) => continue,
                Err(err) => {
                    log::error!("UDS read error; err={err}");
                    return;
                }
            }
        }
    }
}
