//! Create accounts which are later employed to create transactions.
//! Using RpcClient for simplicity.
#![allow(clippy::arithmetic_side_effects)]
use {
    crate::{cli::AccountParams, range::Range},
    futures::future::join_all,
    log::*,
    solana_clock::DEFAULT_MS_PER_SLOT,
    solana_keypair::Keypair,
    solana_message::Message,
    solana_native_token::LAMPORTS_PER_SOL,
    solana_pubkey::Pubkey,
    solana_rpc_client::nonblocking::rpc_client::RpcClient,
    solana_rpc_client_api::{
        client_error::{Error as ClientError, ErrorKind},
        request::RpcError as RequestRpcError,
    },
    solana_signer::Signer,
    solana_system_interface::{instruction as system_instruction, program as system_program},
    solana_transaction::Transaction,
    std::sync::Arc,
    thiserror::Error,
    tokio::time::{sleep, Duration},
};

/// Size of the accounts chunk for which we request accounts info (get_multiple_accounts call).
/// Cannot be greater than 100.
const RPC_ACCOUNT_CHUNK_SIZE: usize = 64;

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

#[derive(Debug)]
pub struct AccountsFile {
    // owner for sized accounts
    pub owner_program_id: Pubkey,
    pub payers: Vec<Keypair>,
    pub sized_accounts: Vec<Keypair>,
}

impl AccountsFile {
    pub async fn validate(
        &self,
        rpc_client: Arc<RpcClient>,
        account_params: AccountParams,
    ) -> Result<bool, AccountsCreatorError> {
        info!("Started validating accounts registry...");
        let program_validation = self.validate_program_deployed(rpc_client.clone());
        let payers_validation = self.validate_payers(
            rpc_client.clone(),
            account_params.num_payers,
            account_params.payer_account_balance,
        );
        let sized_validation = self.validate_sized_accounts(
            rpc_client,
            account_params.num_accounts,
            account_params.account_size,
            account_params.account_owner,
        );
        let result =
            program_validation.await? && payers_validation.await? && sized_validation.await?;
        if result {
            info!("Successfully validated accounts registry.");
        } else {
            error!("Failed accounts registry validation.");
        }
        Ok(result)
    }
    pub async fn validate_program_deployed(
        &self,
        rpc_client: Arc<RpcClient>,
    ) -> Result<bool, AccountsCreatorError> {
        let account = rpc_client.get_account(&self.owner_program_id).await;
        match account {
            Ok(account) => {
                if account.executable {
                    return Ok(true);
                }
                Ok(false)
            }
            Err(err) => {
                // check if account was not found
                if let ErrorKind::RpcError(RequestRpcError::ForUser(_)) = &*err.kind {
                    return Ok(false);
                }
                Err(AccountsCreatorError::ClientError(err))
            }
        }
    }

    pub async fn validate_payers(
        &self,
        rpc_client: Arc<RpcClient>,
        desired_num: usize,
        desired_balance: u64,
    ) -> Result<bool, AccountsCreatorError> {
        if self.payers.len() < desired_num {
            error!(
                "Insufficient number of payers {}, while expected {}",
                self.payers.len(),
                desired_num
            );
            return Ok(false);
        }
        for payer in &self.payers {
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
    pub async fn validate_sized_accounts(
        &self,
        rpc_client: Arc<RpcClient>,
        desired_num: usize,
        desired_account_size: Range,
        desired_owner: Pubkey,
    ) -> Result<bool, AccountsCreatorError> {
        if self.sized_accounts.len() < desired_num {
            error!(
                "Insufficient number of sized accounts {}, while expected {}",
                self.sized_accounts.len(),
                desired_num
            );
            return Ok(false);
        }
        for accounts_chunk in self.sized_accounts.chunks(RPC_ACCOUNT_CHUNK_SIZE) {
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
                    error!("Size of account {} is unexpected.", pubkeys[index]);
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
