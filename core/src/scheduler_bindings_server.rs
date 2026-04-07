use {
    crate::banking_stage::BankingControlMsg, agave_orchestrator::OrchestratorStream,
    agave_scheduling_utils::handshake, std::path::Path, tokio::sync::mpsc,
};

pub(crate) fn spawn(
    path: &Path,
    session_sender: mpsc::Sender<BankingControlMsg>,
    mut orchestrator_stream: Option<OrchestratorStream>,
) {
    // NB: Panic on start if we can't bind.
    let _ = std::fs::remove_file(path);
    let mut listener = handshake::server::Server::new(path).unwrap();

    std::thread::Builder::new()
        .name("solBindingSrv".to_string())
        .spawn(move || {
            // Signal readiness to orchestrator if present.
            if let Some(stream) = orchestrator_stream.as_mut() {
                use std::io::Write;
                match stream.write_all(&[0x01]) {
                    Ok(()) => info!("Sent readiness signal to orchestrator"),
                    Err(err) => error!("Failed to send orchestrator readiness signal: {err}"),
                }
            }

            loop {
                match listener.accept() {
                    Ok(session) => {
                        if session_sender
                            .blocking_send(BankingControlMsg::External { session })
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(err) => {
                        error!("External scheduler handshake failed; err={err}")
                    }
                };
            }
        })
        .unwrap();
}
