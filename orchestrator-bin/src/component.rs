use {
    nix::{
        sys::signal::{self, Signal},
        unistd::Pid,
    },
    std::{
        future::Future,
        pin::Pin,
        task::{Context, Poll, ready},
    },
    tokio::net::UnixStream as TokioUnixStream,
};

pub(crate) struct Component {
    role: Role,
    pid: Pid,
    stream: TokioUnixStream,
}

impl Component {
    pub(crate) fn new(role: Role, pid: Pid, stream: TokioUnixStream) -> Self {
        Self { role, pid, stream }
    }

    pub(crate) fn shutdown(&self) {
        eprintln!(
            "[orchestrator] sending SIGTERM; role={:?}; pid={}",
            self.role, self.pid
        );
        signal::kill(self.pid, Signal::SIGTERM).unwrap();
    }
}

impl Future for Component {
    type Output = Role;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();

        // Wait for component exit (UDS EOF).
        let _ = ready!(this.stream.poll_read_ready(cx));

        // Return role that exited.
        Poll::Ready(this.role)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Role {
    Validator,
    Scheduler,
}
