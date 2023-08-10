// Sentinel value used to indicate to read from stdin instead of file
pub(crate) const STDIN_TOKEN: &str = "-";

pub(crate) const DEFAULT_FLOOD_PACKETS_PER_PEER_PER_ITERATION: u32 = 10;
pub(crate) const DEFAULT_FLOOD_ITERATION_DELAY_US: u64 = 1_000_000;

pub fn load_configuration(path: &String) -> Result<String, String> {
    match &path[..] {
        STDIN_TOKEN => std::io::read_to_string(std::io::stdin())
            .map_err(|e| format!("Failed to read configuration from stdin: {e}")),
        _ => std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read configuration file {path}: {e}")),
    }
}
