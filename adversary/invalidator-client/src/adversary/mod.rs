use {
    log::*,
    reqwest::header::HeaderValue,
    serde::Serialize,
    solana_adversary::{auth::AuthHeader, fetch_auth_token, send_request_verified},
    solana_keypair::Keypair,
    solana_signer::Signer,
};

pub(super) mod delay_votes;
pub(super) mod drop_turbine_votes;
pub(super) mod flood_unused_port;
pub(super) mod gossip;
pub(super) mod leader_block;
pub(super) mod packet_drop;
pub(super) mod repair;
pub(super) mod replay;
pub(super) mod shred_forwarder;
pub(super) mod tpu;

pub trait Command: Serialize + Sized {
    const RPC_METHOD: &'static str;

    fn send_with_auth(
        &self,
        url: &str,
        rpc_adversary_keypair: &Option<Keypair>,
    ) -> Result<(), String> {
        let auth_header = rpc_adversary_keypair
            .as_ref()
            .map(|keypair| {
                trace!("Authenticating with pubkey={}", keypair.pubkey());
                let token = fetch_auth_token(url)?;
                trace!("Fetched authentication token={token:?}");
                let auth_header = AuthHeader::new_signed(&token, keypair, Self::RPC_METHOD, self)?;
                let auth_header = serde_json::to_value(auth_header)
                    .map_err(|e| e.to_string())?
                    .to_string();
                HeaderValue::from_str(&auth_header).map_err(|e| e.to_string())
            })
            .transpose()?;

        let params = serde_json::json!([self]);
        send_request_verified(url, Self::RPC_METHOD, params, auth_header)?;
        Ok(())
    }
}
