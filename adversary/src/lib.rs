#![allow(clippy::arithmetic_side_effects)]

pub mod adversary_context;
pub mod adversary_feature_set;
pub mod flood_worker;
pub mod gossip;
pub mod repair;

use solana_pubkey::Pubkey;

#[derive(Debug)]
enum PeerIdentifierSanitized {
    Pubkey(Pubkey),
    Ip(std::net::IpAddr),
}

impl TryFrom<&String> for PeerIdentifierSanitized {
    type Error = String;
    fn try_from(source: &String) -> Result<Self, Self::Error> {
        if let Ok(pubkey) = Pubkey::try_from(&source[..]) {
            return Ok(PeerIdentifierSanitized::Pubkey(pubkey));
        }
        Ok(PeerIdentifierSanitized::Ip(source.parse().map_err(
            |e| format!("Failed to parse peer identifier {source}: {e}"),
        )?))
    }
}

pub fn verify_peer_identifier(identifier: &String) -> Result<(), String> {
    PeerIdentifierSanitized::try_from(identifier)?;
    Ok(())
}
