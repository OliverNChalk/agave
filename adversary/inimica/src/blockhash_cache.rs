//! Attack run multiple operations, and so it makes sense to cache the last blockhash, to save on
//! the RPC calls.

use {
    log::warn, parking_lot::Mutex, solana_hash::Hash,
    solana_rpc_client::nonblocking::rpc_client::RpcClient,
    solana_rpc_client_api::client_error::Error as RpcClientError, std::sync::Arc, thiserror::Error,
};

#[derive(Error, Debug)]
pub enum BlockhashCacheError {
    #[error("get_latest_blockhash() failed: {0}")]
    GetLatestBlockhash(RpcClientError),
}

#[derive(Debug, Clone)]
pub struct BlockhashCache {
    last_hash: Arc<Mutex<Hash>>,
}

impl BlockhashCache {
    /// Creates a new [`BlockhashCache`].  Note that it contains a default [`Hash`], you want call
    /// [`BlockhashCache::refresh()`] at least once before the first use.
    pub fn uninitialized() -> Self {
        Self {
            last_hash: Arc::default(),
        }
    }

    pub async fn refresh(&self, rpc_client: &RpcClient) -> Result<(), BlockhashCacheError> {
        let blockhash = rpc_client
            .get_latest_blockhash()
            .await
            .map_err(BlockhashCacheError::GetLatestBlockhash)?;
        let mut last_hash = self.last_hash.lock();
        if *last_hash == blockhash {
            // There are two probable cases why you might be seeing this warning:
            // 1. You are refreshing the blockhash too frequently.  It does not make sense to
            //    refresh more frequently than once every slot.  And you probably want even lower
            //    rate to avoid refreshing within the same slot.
            // 2. The cluster is not making any progress, in which case, this warning could help
            //    debug the consensus issue.
            warn!("`get_latest_blockhash()` returned the same blockhash we've seen before.");
        } else {
            *last_hash = blockhash;
        }
        Ok(())
    }

    pub fn get(&self) -> Hash {
        *self.last_hash.lock()
    }
}
