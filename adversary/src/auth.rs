use {
    http::header::HeaderValue,
    rand::{thread_rng, Rng},
    serde::{Deserialize, Serialize},
    solana_keypair::Keypair,
    solana_pubkey::Pubkey,
    solana_signature::Signature,
    solana_signer::Signer,
};

pub const HTTP_HEADER_FIELD_NAME_INVALIDATOR_AUTH: &str = "X-Invalidator-Auth";

pub const JSON_RPC_AUTH_TOKEN_SIZE: usize = 32;

pub type JsonRpcAuthToken = [u8; JSON_RPC_AUTH_TOKEN_SIZE];

#[derive(Default, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthHeader {
    signature: Signature,
    token: JsonRpcAuthToken,
    nonce: u32,
}

impl AuthHeader {
    pub fn new(token: JsonRpcAuthToken) -> Self {
        Self {
            signature: Signature::default(),
            token,
            nonce: thread_rng().gen(),
        }
    }

    pub fn new_signed<T: serde::Serialize>(
        token: &JsonRpcAuthToken,
        keypair: &Keypair,
        call: &str,
        payload: &T,
    ) -> Result<Self, String> {
        let mut auth_header = AuthHeader::new(*token);
        auth_header.sign_request(keypair, call, payload)?;
        Ok(auth_header)
    }

    pub fn nonce(&self) -> u32 {
        self.nonce
    }

    pub fn signature(&self) -> &Signature {
        &self.signature
    }

    pub fn token(&self) -> JsonRpcAuthToken {
        self.token
    }

    pub fn sign_request<T: serde::Serialize>(
        &mut self,
        keypair: &Keypair,
        call: &str,
        payload: &T,
    ) -> Result<(), String> {
        self.signature = Signature::default();
        let header_str = serde_json::to_value(self.clone())
            .map_err(|e| e.to_string())?
            .to_string();
        let payload_str = serde_json::to_value(payload)
            .map_err(|e| e.to_string())?
            .to_string();
        let signable_bytes =
            [call.as_ref(), header_str.as_bytes(), payload_str.as_bytes()].concat();
        self.signature = keypair.sign_message(&signable_bytes[..]);
        Ok(())
    }

    pub fn verify_signature_serialized_payload(
        &self,
        id: &Pubkey,
        call: &str,
        payload_bytes: &[u8],
    ) -> Result<(), String> {
        let mut signed_header = self.clone();
        signed_header.signature = Signature::default();
        let header_str = serde_json::to_value(&signed_header)
            .map_err(|e| e.to_string())?
            .to_string();
        let signed_bytes = [call.as_ref(), header_str.as_bytes(), payload_bytes].concat();
        if !self.signature().verify(id.as_ref(), &signed_bytes) {
            return Err("bad signature".to_string());
        }
        Ok(())
    }

    pub fn to_header_value(&self) -> Option<HeaderValue> {
        let header = serde_json::to_value(self).ok()?.to_string();
        HeaderValue::from_str(&header[..]).ok()
    }
}
