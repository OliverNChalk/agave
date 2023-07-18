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

pub mod drop_turbine_votes;
pub mod leader_block;
pub mod repair;
pub mod shred_forwarder;

pub trait Command: Serialize {
    const RPC_METHOD: &'static str;

    fn send(&self, url: &str) {
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

        let response = match client.post(url).json(&payload).send() {
            Ok(response) => response,
            Err(err) => {
                error!("RPC Send Error: {err:?}");
                return;
            }
        };
        let response = match response.json::<serde_json::Value>() {
            Ok(response) => response,
            Err(err) => {
                error!("RPC Parse Response Error: {err:?}");
                return;
            }
        };
        trace!("rpc command response: {response:#?}");
    }
}
