use {serde::Deserialize, std::path::PathBuf};

#[derive(Debug, Deserialize)]
pub(crate) struct Config {
    pub(crate) orchestrator: OrchestratorConfig,
    pub(crate) scheduler: SchedulerConfig,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OrchestratorConfig {
    pub(crate) bin: PathBuf,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SchedulerConfig {
    pub(crate) bin: PathBuf,
    pub(crate) config: Option<PathBuf>,
}
