use {
    log::*,
    reqwest::blocking::Client,
    serde::{Deserialize, Serialize},
};

#[derive(Debug, Serialize, Deserialize)]
struct RpcRequest {
    jsonrpc: String,
    method: String,
    params: serde_json::Value,
    id: u64,
}

pub(super) mod drop_turbine_votes;
pub(super) mod gossip;
pub(super) mod leader_block;
pub(super) mod packet_drop;
pub(super) mod repair;
pub(super) mod shred_forwarder;

pub trait Command: Serialize {
    const RPC_METHOD: &'static str;

    fn send(&self, url: &str) -> Result<(), String> {
        let params = serde_json::json!([self]);
        let payload = RpcRequest {
            jsonrpc: "2.0".to_string(),
            method: Self::RPC_METHOD.to_string(),
            params,
            id: 1,
        };
        let client = Client::new();
        info!("sending rpc command: {}", Self::RPC_METHOD);
        trace!("rpc command payload: {payload:#?}");
        let response = client
            .post(url)
            .json(&payload)
            .send()
            .map_err(|e| format!("RPC Send Error: {e:?}"))?;
        let response = response
            .json::<serde_json::Value>()
            .map_err(|e| format!("RPC Parse Response Error: {e:?}"))?;
        trace!("rpc command response: {response:#?}");
        Ok(())
    }
}
