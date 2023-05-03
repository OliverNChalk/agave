use {
    crate::{adversary_feature_set::repair_minimal_packet_flood, repair::RepairMinimalPacketFlood},
    solana_validator_exit::Exit,
    std::sync::{LazyLock, RwLock},
};

#[derive(Default)]
pub struct AdversaryContext {
    pub repair_minimal_packet_flood: RwLock<Option<RepairMinimalPacketFlood>>,
}

impl AdversaryContext {
    pub fn register_cleanup(validator_exit: &RwLock<Exit>) {
        validator_exit
            .write()
            .unwrap()
            .register_exit(Box::new(move || {
                let mut adversary_repair = ADVERSARY_CONTEXT
                    .repair_minimal_packet_flood
                    .write()
                    .unwrap();
                repair_minimal_packet_flood::set_config(
                    repair_minimal_packet_flood::AdversarialConfig::default(),
                );
                if let Some(context) = adversary_repair.take() {
                    context.join().unwrap();
                }
            }));
    }
}

pub static ADVERSARY_CONTEXT: LazyLock<AdversaryContext> = LazyLock::new(|| AdversaryContext {
    repair_minimal_packet_flood: RwLock::new(None),
});
