use {
    crate::{
        adversary_feature_set::{
            flood_unused_port, gossip_packet_flood, repair_packet_flood, tpu_packet_flood,
        },
        flood_worker::AdversaryWorkersContext,
    },
    solana_validator_exit::Exit,
    std::sync::{LazyLock, RwLock},
};

#[derive(Default)]
pub struct AdversaryContext {
    pub gossip_packet_flood: RwLock<Option<AdversaryWorkersContext>>,
    pub repair_packet_flood: RwLock<Option<AdversaryWorkersContext>>,
    pub tpu_packet_flood: RwLock<Option<AdversaryWorkersContext>>,
    pub unused_port_packet_flood: RwLock<Option<AdversaryWorkersContext>>,
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

                let mut adversary_gossip = ADVERSARY_CONTEXT.gossip_packet_flood.write().unwrap();
                gossip_packet_flood::set_config(gossip_packet_flood::AdversarialConfig::default());
                if let Some(context) = adversary_gossip.take() {
                    context.join().unwrap();
                }

                let mut adversary_tpu = ADVERSARY_CONTEXT.tpu_packet_flood.write().unwrap();
                tpu_packet_flood::set_config(tpu_packet_flood::AdversarialConfig::default());
                if let Some(context) = adversary_tpu.take() {
                    context.join().unwrap();
                }

                let mut adversary_unused_port =
                    ADVERSARY_CONTEXT.unused_port_packet_flood.write().unwrap();
                flood_unused_port::set_config(flood_unused_port::AdversarialConfig::default());
                if let Some(context) = adversary_unused_port.take() {
                    context.join().unwrap();
                }
            }));
    }
}

pub static ADVERSARY_CONTEXT: LazyLock<AdversaryContext> = LazyLock::new(|| AdversaryContext {
    gossip_packet_flood: RwLock::new(None),
    repair_packet_flood: RwLock::new(None),
    tpu_packet_flood: RwLock::new(None),
    unused_port_packet_flood: RwLock::new(None),
});
