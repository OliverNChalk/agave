use {
    futures::FutureExt,
    nix::{
        sys::signal::{self, Signal},
        unistd::Pid,
    },
    std::{
        future::Future,
        pin::Pin,
        process::ExitStatus,
        task::{Context, Poll, ready},
    },
    tokio::{net::UnixStream as TokioUnixStream, process::Child as TokioChild},
};

pub(crate) struct Component {
    role: Role,
    child: TokioChild,
    stream: TokioUnixStream,
}

impl Component {
    pub(crate) fn new(role: Role, child: TokioChild, stream: TokioUnixStream) -> Self {
        Self {
            role,
            child,
            stream,
        }
    }

    pub(crate) fn shutdown(&self) {
        let pid = self
            .child
            .id()
            .expect("shutdown should not be called after future completes");
        log::info!("Sending SIGTERM; role={:?}; pid={pid}", pid,);
        signal::kill(Pid::from_raw(pid.try_into().unwrap()), Signal::SIGTERM).unwrap();
    }
}

impl Future for Component {
    type Output = (Role, ExitStatus);

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();

        // Wait for component exit (UDS EOF).
        let _ = ready!(this.stream.poll_read_ready(cx));

        // Reap child.
        let code = ready!(Box::pin(this.child.wait()).poll_unpin(cx)).unwrap();

        // Return role that exited.
        Poll::Ready((this.role, code))
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Role {
    Scheduler,
}
