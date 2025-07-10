//! Checkout the `README.md` for the guidance.
use {
    log::*,
    solana_keypair::Keypair,
    solana_rpc_client::nonblocking::rpc_client::RpcClient,
    solana_signer::{EncodableKey, Signer},
    solana_state_loader::{
        accounts_file::{create_file_persisted_accounts, read_accounts_file, AccountsFile},
        cli::{build_cli_parameters, Command, StateLoaderCliParameters},
        error::StateLoaderError,
    },
    std::sync::Arc,
};

#[tokio::main]
async fn main() -> Result<(), StateLoaderError> {
    solana_logger::setup_with_default("solana=info");
    let StateLoaderCliParameters {
        json_rpc_url,
        commitment_config,
        authority,
        validate_accounts,
        command,
    } = build_cli_parameters();
    let authority = if let Some(authority_file) = authority {
        Keypair::read_from_file(authority_file)
            .map_err(|_err| StateLoaderError::KeypairReadFailure)?
    } else {
        // create authority just for this run
        Keypair::new()
    };
    info!("Using authority {}", authority.pubkey());

    let rpc_client = Arc::new(RpcClient::new_with_commitment(
        json_rpc_url.to_string(),
        commitment_config,
    ));

    match command {
        Command::WriteAccounts(write_accounts) => {
            create_file_persisted_accounts(
                rpc_client.clone(),
                authority,
                write_accounts,
                validate_accounts,
            )
            .await?;
        }
        Command::ReadAccounts(read_accounts) => {
            let AccountsFile {
                payers,
                sized_accounts,
                owner_program_id,
            } = read_accounts_file(read_accounts.accounts_file.clone());
            info!(
                "Loaded {} payers and {} sized accounts (owner: {}) from file: {:?}",
                payers.len(),
                sized_accounts.len(),
                owner_program_id,
                read_accounts.accounts_file
            );
        }
    }

    Ok(())
}
