//! Checkout the `README.md` for the guidance.
use {
    log::*,
    solana_cli_config::ConfigInput,
    solana_keypair::Keypair,
    solana_pubkey::Pubkey,
    solana_rpc_client::nonblocking::rpc_client::RpcClient,
    solana_signer::{EncodableKey, Signer},
    solana_streamer::nonblocking::quic::{compute_max_allowed_uni_streams, ConnectionPeerType},
    solana_transaction_bench::{
        accounts_creator::AccountsCreator,
        blockhash_updater::BlockhashUpdater,
        cli::{build_cli_parameters, ClientCliParameters},
        error::BenchClientError,
        generator::TransactionGenerator,
        network::{leader_updater::create_leader_updater, ConnectionWorkersScheduler},
    },
    std::{fmt::Debug, sync::Arc},
    tokio::{
        sync::{mpsc, watch},
        task::JoinHandle,
    },
    tokio_util::sync::CancellationToken,
};

/// Specifies the number of slots ahead to establish connections with scheduled peers.
/// This constant determines how far in advance connections are created, not the number
/// of peers to which transactions are sent.
const LOOKAHEAD_SLOTS: u64 = 8;
const GENERATOR_CHANNEL_SIZE: usize = 32;

fn main() {
    solana_logger::setup_with_default("solana=info");

    let opt = build_cli_parameters();
    let code = {
        if let Err(e) = run(opt) {
            error!("ERROR: {e}");
            1
        } else {
            0
        }
    };
    ::std::process::exit(code);
}

#[tokio::main]
async fn run(parameters: ClientCliParameters) -> Result<(), BenchClientError> {
    let validator_identity = if let Some(staked_identity_file) = parameters.staked_identity_file {
        Some(
            Keypair::read_from_file(staked_identity_file)
                .map_err(|_err| BenchClientError::KeypairReadFailure)?,
        )
    } else {
        None
    };

    let authority = if let Some(authority_file) = parameters.authority {
        Keypair::read_from_file(authority_file)
            .map_err(|_err| BenchClientError::KeypairReadFailure)?
    } else {
        // create authority just for this run
        Keypair::new()
    };
    info!("Use authority {}", authority.pubkey());

    let (_, websocket_url) =
        ConfigInput::compute_websocket_url_setting("", "", &parameters.json_rpc_url, "");

    let rpc_client = Arc::new(RpcClient::new_with_commitment(
        parameters.json_rpc_url.to_string(),
        parameters.commitment_config,
    ));

    let accounts_creator =
        AccountsCreator::new(rpc_client.clone(), authority, parameters.account_params);
    let accounts = accounts_creator.create().await?;
    if parameters.validate_accounts
        && !accounts
            .validate(rpc_client.clone(), parameters.account_params)
            .await?
    {
        return Err(BenchClientError::AccountsValidationFailure);
    }

    // Set up size of the txs batch to put into the queue to be equal to the num_streams_per_connection
    let num_streams_per_connection = compute_num_streams(
        &rpc_client,
        validator_identity.as_ref().map(|keypair| keypair.pubkey()),
    )
    .await?;
    let send_batch_size = num_streams_per_connection;
    info!("Number of streams per connection is {num_streams_per_connection}.");

    let blockhash = rpc_client
        .get_latest_blockhash()
        .await
        .expect("Blockhash request should not fail.");
    let (blockhash_sender, blockhash_receiver) = watch::channel(blockhash);
    let blockhash_updater = BlockhashUpdater::new(rpc_client.clone(), blockhash_sender);

    let blockhash_task_handle = tokio::spawn(async move { blockhash_updater.run().await });

    // Use bounded to avoid producing too many batches of transactions.
    let (transaction_sender, transaction_receiver) = mpsc::channel(GENERATOR_CHANNEL_SIZE);

    let transaction_generator = TransactionGenerator::new(
        accounts,
        blockhash_receiver,
        transaction_sender,
        parameters.transaction_params,
        send_batch_size,
        parameters.duration,
    );

    let transaction_generator_task_handle =
        tokio::spawn(async move { transaction_generator.run().await });

    let scheduler_handle: JoinHandle<Result<(), BenchClientError>> = tokio::spawn(async move {
        let leader_updater = create_leader_updater(
            rpc_client,
            websocket_url,
            LOOKAHEAD_SLOTS,
            parameters.pinned_address,
        )
        .await?;

        // TODO It would be good to connect the `cancel` token to the Ctrl+C and other interrupt
        // handlers.
        let cancel = CancellationToken::new();

        ConnectionWorkersScheduler::run(
            parameters.bind,
            validator_identity,
            leader_updater,
            cancel,
            transaction_receiver,
            parameters.num_max_open_connections,
        )
        .await?;
        Ok(())
    });

    join_service(transaction_generator_task_handle, "TransactionGenerator").await;
    join_service(blockhash_task_handle, "BlockhashUpdater").await;
    join_service::<BenchClientError>(scheduler_handle, "Scheduler").await;

    Ok(())
}

// TODO(klykov): this function is taken from bench-tps and rewritten to be async.
// Move it back when bench-tps will become async.
/// Request information about node's stake
/// If fail to get requested information, return error
/// Otherwise return stake of the node
/// along with total activated stake of the network
async fn find_node_activated_stake(
    rpc_client: &Arc<RpcClient>,
    node_id: Option<Pubkey>,
) -> Result<(Option<u64>, u64), BenchClientError> {
    let vote_accounts = rpc_client
        .get_vote_accounts()
        .await
        .map_err(|_| BenchClientError::FindValidatorIdentityFailure)?;

    let total_active_stake: u64 = vote_accounts
        .current
        .iter()
        .map(|vote_account| vote_account.activated_stake)
        .sum();

    let Some(node_id) = node_id else {
        return Ok((None, total_active_stake));
    };
    let node_id_as_str = node_id.to_string();
    let find_result = vote_accounts
        .current
        .iter()
        .find(|&vote_account| vote_account.node_pubkey == node_id_as_str);
    match find_result {
        Some(value) => Ok((Some(value.activated_stake), total_active_stake)),
        None => Err(BenchClientError::FindValidatorIdentityFailure),
    }
}

async fn compute_num_streams(
    rpc_client: &Arc<RpcClient>,
    validator_pubkey: Option<Pubkey>,
) -> Result<usize, BenchClientError> {
    let (validator_stake, total_stake) =
        find_node_activated_stake(rpc_client, validator_pubkey).await?;
    debug!(
        "Validator {validator_pubkey:?} stake: {validator_stake:?}, total stake: {total_stake}."
    );
    let client_type = validator_stake.map_or(ConnectionPeerType::Unstaked, |stake| {
        ConnectionPeerType::Staked(stake)
    });
    Ok(compute_max_allowed_uni_streams(client_type, total_stake))
}

async fn join_service<Error>(handle: JoinHandle<Result<(), Error>>, task_name: &str)
where
    Error: Debug,
{
    match handle.await {
        Ok(Ok(_)) => info!("Task {task_name} completed successfully"),
        Ok(Err(e)) => error!("Task failed with error: {e:?}"),
        Err(e) => error!("Task was cancelled or panicked: {e:?}"),
    }
}
