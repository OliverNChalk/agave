use {
    crate::{
        accounts_creator::AccountsCreatorError, accounts_file::AccountsFile, cli::AccountParams,
        range::Range,
    },
    log::{error, info},
    solana_pubkey::Pubkey,
    solana_rpc_client::nonblocking::rpc_client::RpcClient,
    solana_rpc_client_api::{client_error::ErrorKind, request::RpcError as RequestRpcError},
    solana_signer::Signer,
    std::sync::Arc,
};

/// Size of the accounts chunk for which we request accounts info (get_multiple_accounts call).
/// Cannot be greater than 100.
const RPC_ACCOUNT_CHUNK_SIZE: usize = 64;

pub async fn validate(
    account_file: &AccountsFile,
    rpc_client: Arc<RpcClient>,
    AccountParams {
        num_accounts,
        num_payers,
        account_size,
        payer_account_balance,
        account_owner,
    }: AccountParams,
) -> Result<bool, AccountsCreatorError> {
    info!("Started validating accounts registry...");
    let program_validation = validate_program_deployed(account_file, rpc_client.clone());
    let payers_validation = validate_payers(
        account_file,
        rpc_client.clone(),
        num_payers,
        payer_account_balance,
    );
    let sized_validation = validate_sized_accounts(
        account_file,
        rpc_client,
        num_accounts,
        account_size,
        account_owner,
    );
    let result = program_validation.await? && payers_validation.await? && sized_validation.await?;
    if result {
        info!("Successfully validated accounts registry.");
    } else {
        error!("Failed accounts registry validation.");
    }
    Ok(result)
}

async fn validate_program_deployed(
    AccountsFile {
        owner_program_id, ..
    }: &AccountsFile,
    rpc_client: Arc<RpcClient>,
) -> Result<bool, AccountsCreatorError> {
    let account = rpc_client.get_account(owner_program_id).await;
    match account {
        Ok(account) => Ok(account.executable),
        Err(err) => {
            // check if account was not found
            if let ErrorKind::RpcError(RequestRpcError::ForUser(_)) = &*err.kind {
                return Ok(false);
            }
            Err(AccountsCreatorError::ClientError(err))
        }
    }
}

async fn validate_payers(
    AccountsFile { payers, .. }: &AccountsFile,
    rpc_client: Arc<RpcClient>,
    desired_num: usize,
    desired_balance: u64,
) -> Result<bool, AccountsCreatorError> {
    if payers.len() < desired_num {
        error!(
            "Insufficient number of payers {}, while expected {}",
            payers.len(),
            desired_num
        );
        return Ok(false);
    }
    for payer in payers {
        let balance = rpc_client.get_balance(&payer.pubkey()).await?;
        if balance < desired_balance {
            error!(
                "Insufficient balance {} for account {}.",
                balance,
                payer.pubkey()
            );
            return Ok(false);
        }
    }
    Ok(true)
}

async fn validate_sized_accounts(
    AccountsFile { sized_accounts, .. }: &AccountsFile,
    rpc_client: Arc<RpcClient>,
    desired_num: usize,
    desired_account_size: Range,
    desired_owner: Pubkey,
) -> Result<bool, AccountsCreatorError> {
    if sized_accounts.len() < desired_num {
        error!(
            "Insufficient number of sized accounts {}, while expected {}",
            sized_accounts.len(),
            desired_num
        );
        return Ok(false);
    }
    for accounts_chunk in sized_accounts.chunks(RPC_ACCOUNT_CHUNK_SIZE) {
        let pubkeys: Vec<Pubkey> = accounts_chunk
            .iter()
            .map(|keypair| keypair.pubkey())
            .collect();
        let accounts_info = rpc_client.get_multiple_accounts(&pubkeys).await?;
        for (index, info) in accounts_info.iter().enumerate() {
            let Some(info) = info else {
                error!("Account {} doesn't exist.", pubkeys[index]);
                return Ok(false);
            };
            if !desired_account_size.contains(info.data.len()) {
                error!(
                    "Size of account {} is unexpected. Expected range: {:?}",
                    pubkeys[index], desired_account_size
                );
                return Ok(false);
            }
            if info.owner != desired_owner {
                error!(
                    "Unexpected account owner for {}. Expected {}, found {}.",
                    pubkeys[index], desired_owner, info.owner
                );
                return Ok(false);
            }
        }
    }
    Ok(true)
}
