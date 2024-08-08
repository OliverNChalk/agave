//! Create accounts which are later employed to create transactions.
//! Using RpcClient for simplicity.
#![allow(clippy::arithmetic_side_effects)]
use {
    crate::{accounts_file::AccountsFile, cli::AccountParams, range::Range},
    futures::future::join_all,
    log::*,
    solana_clock::DEFAULT_MS_PER_SLOT,
    solana_keypair::Keypair,
    solana_message::Message,
    solana_native_token::LAMPORTS_PER_SOL,
    solana_pubkey::Pubkey,
    solana_rpc_client::nonblocking::rpc_client::RpcClient,
    solana_rpc_client_api::client_error::Error as ClientError,
    solana_signer::Signer,
    solana_system_interface::{instruction as system_instruction, program as system_program},
    solana_transaction::Transaction,
    std::sync::Arc,
    thiserror::Error,
    tokio::time::{sleep, Duration},
};

/// How many transactions send concurrently.
const RPC_SEND_TX_BATCH: usize = 64;
/// Used to sleep between accounts creation to avoid getting 429s from RPC.
const ACCOUNT_CREATION_SLEEP_INTERVAL_MS: Duration = Duration::from_millis(150);

#[derive(Error, Debug)]
pub enum AccountsCreatorError {
    #[error(transparent)]
    ClientError(#[from] ClientError),

    #[error("Failed to airdrop")]
    AirdropFailure,

    #[error("Failed to create account")]
    CreateAccountFailure,
}

pub struct AccountsCreator {
    rpc_client: Arc<RpcClient>,
    authority: Keypair,
    account_params: AccountParams,
}

impl AccountsCreator {
    pub fn new(
        rpc_client: Arc<RpcClient>,
        authority: Keypair,
        account_params: AccountParams,
    ) -> Self {
        Self {
            rpc_client,
            authority,
            account_params,
        }
    }
    pub async fn create(&self) -> Result<AccountsFile, AccountsCreatorError> {
        self.ensure_authority_balance().await?;
        let payers = self.create_payers().await?;
        info!("Payers have been created.");
        let sized_accounts = self.create_sized_accounts(&payers).await?;
        info!("Sized accounts have been created.");
        Ok(AccountsFile {
            owner_program_id: self.account_params.account_owner,
            payers,
            sized_accounts,
        })
    }

    async fn ensure_authority_balance(&self) -> Result<(), AccountsCreatorError> {
        let authority_pubkey = self.authority.pubkey();
        let rpc_client = &*self.rpc_client;

        // Compute the minimum budget for payers
        let min_balance_to_create_account = self.request_create_account_tx_fee(0).await?
            + self.account_params.payer_account_balance * LAMPORTS_PER_SOL;
        let required_balance =
            self.account_params.num_payers as u64 * min_balance_to_create_account;
        let actual_balance = rpc_client.get_balance(&authority_pubkey).await?;
        info!("Authority balance {actual_balance}, min required balance {required_balance}");

        if actual_balance >= required_balance {
            return Ok(());
        }

        // The authority needs more SOL.
        let balance_shortage = required_balance.saturating_sub(actual_balance);
        rpc_client
            .request_airdrop(&authority_pubkey, balance_shortage)
            .await?;

        // TODO(klykov): Wait for two blocks before checking.
        // Maybe it is better to check the status using the returned signature?
        sleep(Duration::from_millis(2 * DEFAULT_MS_PER_SLOT)).await;

        let actual_balance = rpc_client.get_balance(&authority_pubkey).await?;
        info!("Balance after airdrop {actual_balance}");

        if actual_balance < required_balance {
            return Err(AccountsCreatorError::AirdropFailure);
        }

        Ok(())
    }

    /// Computes the fee to create account of given size.
    async fn request_create_account_tx_fee(&self, size: u64) -> Result<u64, AccountsCreatorError> {
        // Create dummy create account transaction message to calculate fee
        let rent = self
            .rpc_client
            .get_minimum_balance_for_rent_exemption(size as usize)
            .await?;
        let payer_pubkey = Pubkey::new_unique();
        let instructions = vec![system_instruction::create_account(
            &payer_pubkey,
            &Pubkey::new_unique(),
            rent,
            size,
            &system_program::id(),
        )];

        let blockhash = self.rpc_client.get_latest_blockhash().await?;
        let message = Message::new_with_blockhash(&instructions, Some(&payer_pubkey), &blockhash);
        let fee = self.rpc_client.get_fee_for_message(&message).await?;
        Ok(fee)
    }

    async fn create_payers(&self) -> Result<Vec<Keypair>, AccountsCreatorError> {
        self.create_accounts(
            &[self.authority.insecure_clone()],
            self.account_params.num_payers,
            Range { min: 0, max: 0 },
            self.account_params.payer_account_balance * LAMPORTS_PER_SOL,
            system_program::id(),
        )
        .await
    }
    async fn create_sized_accounts(
        &self,
        payers: &[Keypair],
    ) -> Result<Vec<Keypair>, AccountsCreatorError> {
        let rent_exempt = self
            .rpc_client
            .get_minimum_balance_for_rent_exemption(self.account_params.account_size.max)
            .await?;
        info!("Rent exempt amount is {rent_exempt}.");
        self.create_accounts(
            payers,
            self.account_params.num_accounts,
            self.account_params.account_size,
            rent_exempt,
            self.account_params.account_owner,
        )
        .await
    }
    async fn create_accounts(
        &self,
        authorities: &[Keypair],
        num_accounts: usize,
        account_size: Range,
        balance: u64,
        account_owner: Pubkey,
    ) -> Result<Vec<Keypair>, AccountsCreatorError> {
        // It makes sense to send concurrently subset
        // of transactions to avoid having expired block height exceed error.
        // Take into account that the total size of allocated memory in
        // the block is limited by MAX_BLOCK_ACCOUNTS_DATA_SIZE_DELTA
        // which is ~100MB on the moment of writing.

        let mut authorities_iter = authorities.iter().cycle();
        let mut accounts = Vec::with_capacity(num_accounts);

        let batch_size = std::cmp::min(num_accounts, RPC_SEND_TX_BATCH);
        let total_batches = usize::div_ceil(num_accounts, batch_size);
        for ibatch in 0..total_batches {
            let num_accounts_left = num_accounts - ibatch * batch_size;
            let current_batch_size = std::cmp::min(batch_size, num_accounts_left);

            let authority = authorities_iter
                .next()
                .expect("Authorities slice should not be empty.");

            //1. create batch of N transactions
            let blockhash = self.rpc_client.get_latest_blockhash().await?;
            let keypair_batch: Vec<Keypair> =
                (0..current_batch_size).map(|_| Keypair::new()).collect();
            let txs = keypair_batch.iter().map(|new_account| {
                let instructions = vec![system_instruction::create_account(
                    &authority.pubkey(),
                    &new_account.pubkey(),
                    balance,
                    account_size.uniform() as u64,
                    &account_owner,
                )];

                let message = Message::new(&instructions, Some(&authority.pubkey()));
                Transaction::new(&[authority, new_account], message, blockhash)
            });

            //2. send them concurrently to RPC with confirmation
            let futures = txs
                .map(|tx| async move { self.rpc_client.send_and_confirm_transaction(&tx).await });
            //3. check how many landed
            let results = join_all(futures).await;
            for result in results {
                if let Err(err) = result {
                    debug!("Error {err}");
                    return Err(AccountsCreatorError::CreateAccountFailure);
                }
            }
            accounts.extend(keypair_batch);
            sleep(ACCOUNT_CREATION_SLEEP_INTERVAL_MS).await;
        }
        Ok(accounts)
    }
}
