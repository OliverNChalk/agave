#![allow(clippy::arithmetic_side_effects)]

use {
    crate::auth::{JsonRpcAuthToken, HTTP_HEADER_FIELD_NAME_INVALIDATOR_AUTH},
    accounts_file::AccountsFile,
    adversary_feature_set::replay_stage_attack,
    crossbeam_channel::{Receiver, Sender},
    log::*,
    reqwest::{
        blocking::Client,
        header::{HeaderMap, HeaderValue},
    },
    serde::{Deserialize, Serialize},
    serde_json::Value,
    solana_pubkey::Pubkey,
    std::{fmt, sync::Arc},
};

pub mod accounts_file;
pub mod adversary_context;
pub mod adversary_feature_set;
pub mod auth;
pub mod block_generator_config;
pub mod flood_worker;
pub mod gossip;
pub mod repair;
pub mod tpu;

/// `Sender` and `Receiver` types to exchange information about requested attacks
/// between rpc, banking stage and attach scheduler.
pub type ReplayAttackSender = Sender<SelectedReplayAttack>;
pub type ReplayAttackReceiver = Receiver<SelectedReplayAttack>;

/// Selected attack targeting replay stage.
///
/// If `None`, invalidator performs normal banking stage processing.
/// Otherwise, it builds blocks using specified attack type and accounts.
pub enum SelectedReplayAttack {
    None,
    Selected {
        attack: replay_stage_attack::Attack,
        accounts: Arc<AccountsFile>,
    },
}

impl fmt::Debug for SelectedReplayAttack {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SelectedReplayAttack::None => write!(fmt, "SelectedReplayAttack::None"),
            SelectedReplayAttack::Selected {
                attack,
                accounts: _,
            } => fmt
                .debug_struct("SelectedReplayAttack::Selected")
                .field("attack", attack)
                .finish_non_exhaustive(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RpcRequest {
    pub jsonrpc: String,
    pub method: String,
    pub params: Value,
    pub id: u64,
}

pub fn send_request(
    url: &str,
    method: &str,
    params: Value,
    auth_header: Option<HeaderValue>,
) -> Result<Value, String> {
    let payload = RpcRequest {
        jsonrpc: "2.0".to_string(),
        method: method.to_string(),
        params,
        id: 1,
    };

    let mut headers = HeaderMap::default();
    if let Some(auth_header) = auth_header {
        headers.insert(HTTP_HEADER_FIELD_NAME_INVALIDATOR_AUTH, auth_header);
    }

    let client = Client::new();
    info!("sending rpc command: {method} to {url}");
    trace!("rpc command payload: {payload:#?}");
    let response = client
        .post(url)
        .headers(headers)
        .json(&payload)
        .send()
        .map_err(|e| format!("RPC Send Error: {e:?}"))?;
    trace!("rpc command response: {response:#?}");
    let response = response
        .json::<Value>()
        .map_err(|e| format!("RPC Parse Response Error: {e:?}"))?;
    trace!("rpc command json: {response:#?}");
    Ok(response)
}

pub fn send_request_verified(
    url: &str,
    method: &str,
    params: Value,
    auth_header: Option<HeaderValue>,
) -> Result<Value, String> {
    let response = send_request(url, method, params, auth_header)?;
    if let serde_json::Value::Object(response_map) = &response {
        if response_map.contains_key("error") {
            return Err(response_map["error"].to_string());
        }
    }
    Ok(response)
}

pub fn fetch_auth_token(url: &str) -> Result<JsonRpcAuthToken, String> {
    let response = send_request(url, "fetchAuthToken", Value::Null, None)?;
    let Some(result) = response.get("result") else {
        return Err("failed to parse response".to_string());
    };
    let token: JsonRpcAuthToken =
        serde_json::from_value(result.clone()).map_err(|e| format!("{e}"))?;
    Ok(token)
}

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
