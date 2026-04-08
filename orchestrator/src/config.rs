use {crate::SchedulerTopology, serde::Deserialize, std::path::PathBuf};

#[derive(Debug, Deserialize)]
pub struct Config {
    pub topology: TopologyConfig,
    pub orchestrator: OrchestratorConfig,
    pub scheduler: SchedulerConfig,
}

#[derive(Debug, Deserialize)]
pub struct TopologyConfig {
    pub scheduler: SchedulerTopology,
}

#[derive(Debug, Deserialize)]
pub struct OrchestratorConfig {
    pub bin: PathBuf,
    pub log: PathBuf,
}

#[derive(Debug, Deserialize)]
pub struct SchedulerConfig {
    pub bin: PathBuf,
    pub log: PathBuf,
    pub config: Option<PathBuf>,
}
