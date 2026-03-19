use crate::TelemetryStamp;

pub fn try_open(_: &str) -> Option<shaq::Producer<TelemetryStamp>> {
    None
}
