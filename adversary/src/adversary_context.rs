use {
    crate::repair::RepairMinimalPacketFlood,
    std::sync::{LazyLock, RwLock},
};

#[derive(Default)]
pub struct AdversaryContext {
    pub repair_minimal_packet_flood: RwLock<Option<RepairMinimalPacketFlood>>,
}

pub static ADVERSARY_CONTEXT: LazyLock<AdversaryContext> = LazyLock::new(|| AdversaryContext {
    repair_minimal_packet_flood: RwLock::new(None),
});
