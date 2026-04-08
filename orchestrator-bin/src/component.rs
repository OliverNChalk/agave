use {
    std::{
        future::Future,
        pin::Pin,
        task::{Context, Poll, ready},
    },
    tokio::{io::AsyncWriteExt, net::UnixStream as TokioUnixStream},
};

pub(crate) struct Component {
    role: Role,
    stream: TokioUnixStream,
}

impl Component {
    pub(crate) fn new(role: Role, stream: TokioUnixStream) -> Self {
        Self { role, stream }
    }

    pub(crate) async fn shutdown(&mut self) {
        self.stream.shutdown().await.unwrap();
    }
}

impl Future for Component {
    type Output = Role;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();

        // Wait for component exit.
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
