//! Provides a helper function for bypassing normal BankingStage when a
//! generating attack is running.
//!

// todo Faycel: fixme

// use {
//     crate::banking_stage::immutable_deserialized_packet::PacketFilterFailure,
//     agave_banking_stage_ingress_types::BankingPacketReceiver,
//     crossbeam_channel::RecvTimeoutError,
//     std::{
//         sync::{
//             atomic::{AtomicBool, Ordering},
//             Arc,
//         },
//         time::Duration,
//     },
// };

// Drop packets for a short duration when a generating attack is running.
// If an attack is not running, this function does nothing.
// pub fn drop_loop(receiver: &BankingPacketReceiver, drop_packets: Arc<AtomicBool>) {
//     // Receive and drop up to `CAPACITY_PER_ITERATION` packets every `DROP_DURATION`.
//     const DROP_DURATION: Duration = Duration::from_millis(100);
//     const CAPACITY_PER_ITERATION: usize = 1024;

//     while drop_packets.load(Ordering::Relaxed) {
//         // `packet_filter` is set to drop everything, so the returned `packets` should really be
//         // empty.
//         let mut packet_deserializer = PacketDeserializer::new(receiver.clone());
//         match packet_deserializer.receive_packets_with_filter(
//             DROP_DURATION,
//             CAPACITY_PER_ITERATION,
//             |_| Err(PacketFilterFailure::AttackIsActive),
//         ) {
//             Ok(packets) => drop(packets),
//             Err(RecvTimeoutError::Timeout) => {}
//             Err(RecvTimeoutError::Disconnected) => break,
//         }
//     }
// }
