use {
    bincode::{serialize, Result},
    solana_clock::Slot,
    solana_gossip::ping_pong,
    solana_hash::{Hash, HASH_BYTES},
    solana_keypair::Keypair,
    solana_ledger::shred::{Nonce, SIZE_OF_NONCE},
    solana_packet::PACKET_DATA_SIZE,
    solana_pubkey::{Pubkey, PUBKEY_BYTES},
    solana_signature::{Signature, SIGNATURE_BYTES},
    solana_signer::Signer,
};

/// the number of slots to respond with when responding to `Orphan` requests
pub const MAX_ORPHAN_REPAIR_RESPONSES: usize = 11;

#[cfg_attr(feature = "frozen-abi", derive(AbiExample))]
#[derive(Debug, Deserialize, Serialize)]
pub struct RepairRequestHeader {
    pub signature: Signature,
    pub sender: Pubkey,
    pub recipient: Pubkey,
    pub timestamp: u64,
    pub nonce: Nonce,
}

impl RepairRequestHeader {
    pub fn new(sender: Pubkey, recipient: Pubkey, timestamp: u64, nonce: Nonce) -> Self {
        Self {
            signature: Signature::default(),
            sender,
            recipient,
            timestamp,
            nonce,
        }
    }
}

/// Number of bytes in the randomly generated token sent with ping messages.
pub const REPAIR_PING_TOKEN_SIZE: usize = HASH_BYTES;
pub const REPAIR_RESPONSE_SERIALIZED_PING_BYTES: usize =
    4 /*enum discriminator*/ + PUBKEY_BYTES + REPAIR_PING_TOKEN_SIZE + SIGNATURE_BYTES;

pub type Ping = ping_pong::Ping<REPAIR_PING_TOKEN_SIZE>;
pub type PingCache = ping_pong::PingCache<REPAIR_PING_TOKEN_SIZE>;

/// Window protocol messages
#[cfg_attr(
    feature = "frozen-abi",
    derive(AbiEnumVisitor, AbiExample),
    frozen_abi(digest = "BCJ17nCJHrDK65Qe1CsYucrCUaaDbWsRdkUAdpGkPbyo")
)]
#[derive(Debug, Deserialize, Serialize)]
pub enum RepairProtocol {
    LegacyWindowIndex,
    LegacyHighestWindowIndex,
    LegacyOrphan,
    LegacyWindowIndexWithNonce,
    LegacyHighestWindowIndexWithNonce,
    LegacyOrphanWithNonce,
    LegacyAncestorHashes,
    Pong(ping_pong::Pong),
    WindowIndex {
        header: RepairRequestHeader,
        slot: Slot,
        shred_index: u64,
    },
    HighestWindowIndex {
        header: RepairRequestHeader,
        slot: Slot,
        shred_index: u64,
    },
    Orphan {
        header: RepairRequestHeader,
        slot: Slot,
    },
    AncestorHashes {
        header: RepairRequestHeader,
        slot: Slot,
    },
}

impl RepairProtocol {
    pub fn sender(&self) -> Option<&Pubkey> {
        match self {
            Self::LegacyWindowIndex
            | Self::LegacyHighestWindowIndex
            | Self::LegacyOrphan
            | Self::LegacyWindowIndexWithNonce
            | Self::LegacyHighestWindowIndexWithNonce
            | Self::LegacyOrphanWithNonce
            | Self::LegacyAncestorHashes => None,
            Self::Pong(pong) => Some(pong.from()),
            Self::WindowIndex { header, .. } => Some(&header.sender),
            Self::HighestWindowIndex { header, .. } => Some(&header.sender),
            Self::Orphan { header, .. } => Some(&header.sender),
            Self::AncestorHashes { header, .. } => Some(&header.sender),
        }
    }

    pub fn supports_signature(&self) -> bool {
        match self {
            Self::LegacyWindowIndex
            | Self::LegacyHighestWindowIndex
            | Self::LegacyOrphan
            | Self::LegacyWindowIndexWithNonce
            | Self::LegacyHighestWindowIndexWithNonce
            | Self::LegacyOrphanWithNonce
            | Self::LegacyAncestorHashes => false,
            Self::Pong(_)
            | Self::WindowIndex { .. }
            | Self::HighestWindowIndex { .. }
            | Self::Orphan { .. }
            | Self::AncestorHashes { .. } => true,
        }
    }

    fn max_response_packets(&self) -> usize {
        match self {
            RepairProtocol::WindowIndex { .. }
            | RepairProtocol::HighestWindowIndex { .. }
            | RepairProtocol::AncestorHashes { .. } => 1,
            RepairProtocol::Orphan { .. } => MAX_ORPHAN_REPAIR_RESPONSES,
            RepairProtocol::Pong(_) => 0, // no response
            RepairProtocol::LegacyWindowIndex
            | RepairProtocol::LegacyHighestWindowIndex
            | RepairProtocol::LegacyOrphan
            | RepairProtocol::LegacyWindowIndexWithNonce
            | RepairProtocol::LegacyHighestWindowIndexWithNonce
            | RepairProtocol::LegacyOrphanWithNonce
            | RepairProtocol::LegacyAncestorHashes => 0, // unsupported
        }
    }

    pub fn max_response_bytes(&self) -> usize {
        self.max_response_packets() * PACKET_DATA_SIZE
    }

    pub fn repair_proto_to_bytes(&self, keypair: &Keypair) -> Result<Vec<u8>> {
        debug_assert!(self.supports_signature());
        let mut payload = serialize(self)?;
        let signable_data = [&payload[..4], &payload[4 + SIGNATURE_BYTES..]].concat();
        let signature = keypair.sign_message(&signable_data[..]);
        payload[4..4 + SIGNATURE_BYTES].copy_from_slice(signature.as_ref());
        Ok(payload)
    }
}

#[cfg_attr(
    feature = "frozen-abi",
    derive(AbiEnumVisitor, AbiExample),
    frozen_abi(digest = "9A6ae44qpdT7PaxiDZbybMM2mewnSnPs3C4CxhpbbYuV")
)]
#[derive(Debug, Deserialize, Serialize)]
pub enum RepairResponse {
    Ping(Ping),
}

pub const MAX_ANCESTOR_BYTES_IN_PACKET: usize =
    PACKET_DATA_SIZE -
    SIZE_OF_NONCE -
    4 /*(response version enum discriminator)*/ -
    4 /*slot_hash length*/;
pub const MAX_ANCESTOR_RESPONSES: usize =
    MAX_ANCESTOR_BYTES_IN_PACKET / std::mem::size_of::<(Slot, Hash)>();

#[cfg_attr(
    feature = "frozen-abi",
    derive(AbiEnumVisitor, AbiExample),
    frozen_abi(digest = "GPS6e6pgUdbXLwXN6XHTqrUVMwAL2YKLPDawgMi5hHzi")
)]
#[derive(Debug, Deserialize, Serialize)]
pub enum AncestorHashesResponse {
    Hashes(Vec<(Slot, Hash)>),
    Ping(Ping),
}
