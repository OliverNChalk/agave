use {
    crate::{adversary_feature_set::repair_packet_flood, flood_worker::AdversaryWorkersContext},
    solana_validator_exit::Exit,
    std::sync::{LazyLock, RwLock},
};

#[derive(Default)]
pub struct AdversaryContext {
    pub gossip_packet_flood: RwLock<Option<AdversaryWorkersContext>>,
    pub repair_packet_flood: RwLock<Option<AdversaryWorkersContext>>,
    pub tpu_packet_flood: RwLock<Option<AdversaryWorkersContext>>,
}

impl AdversaryContext {
    pub fn register_cleanup(validator_exit: &RwLock<Exit>) {
        validator_exit
            .write()
            .unwrap()
            .register_exit(Box::new(move || {
                let mut adversary_repair = ADVERSARY_CONTEXT.repair_packet_flood.write().unwrap();
                repair_packet_flood::set_config(repair_packet_flood::AdversarialConfig::default());
                if let Some(context) = adversary_repair.take() {
                    context.join().unwrap();
                }
            }));
    }
}

pub static ADVERSARY_CONTEXT: LazyLock<AdversaryContext> = LazyLock::new(|| AdversaryContext {
    gossip_packet_flood: RwLock::new(None),
    repair_packet_flood: RwLock::new(None),
    tpu_packet_flood: RwLock::new(None),
});
