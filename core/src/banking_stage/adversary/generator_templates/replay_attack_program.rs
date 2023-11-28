//! A common implementation for transaction generators that execute accounts access programs

use {
    solana_adversary::{
        accounts_file::AccountsFile,
        adversary_feature_set::replay_stage_attack::AttackProgramConfig,
    },
    solana_compute_budget::compute_budget_limits::MAX_COMPUTE_UNIT_LIMIT,
    solana_instruction::AccountMeta,
    solana_keypair::Keypair,
    solana_message::Message,
    solana_pubkey::Pubkey,
    solana_runtime::bank::Bank,
    solana_signer::Signer,
    solana_transaction::{sanitized::SanitizedTransaction, Transaction},
    std::sync::Arc,
};

pub(crate) fn verify(accounts: &AccountsFile, attack: AttackProgramConfig) -> Result<(), String> {
    let AttackProgramConfig {
        transaction_batch_size,
        num_accounts_per_tx,
        transaction_cu_budget,
        ..
    } = attack;

    let accounts_batch_size = transaction_batch_size * num_accounts_per_tx;

    let payers_len = accounts.payers.len();
    if payers_len < accounts_batch_size {
        return Err(format!(
            "Not enough \"payer\" accounts: need at least {accounts_batch_size}\n\"payer\" \
             accounts: {payers_len}"
        ));
    }

    let num_max_size_accounts = accounts.max_size.len();
    if num_max_size_accounts < accounts_batch_size {
        return Err(format!(
            "Accounts batch size (`transaction_batch_size` * `num_accounts_per_tx`) must be less \
             than the number of \"max_size\" accounts.\nAccounts batch size: \
             {accounts_batch_size}\n\"max_size\" accounts: {num_max_size_accounts}",
        ));
    }

    if transaction_batch_size == 0 || transaction_batch_size > 64 {
        return Err(format!(
            "`transaction_batch_size` ({transaction_batch_size}) must be in range [1, 64]",
        ));
    }
    if num_accounts_per_tx == 0 || num_accounts_per_tx > 48 {
        return Err(format!(
            "`num_accounts_per_tx` ({num_accounts_per_tx}) must be in range [1, 48]",
        ));
    }
    if transaction_cu_budget > MAX_COMPUTE_UNIT_LIMIT {
        return Err(format!(
            "`transaction_cu_budget` ({transaction_cu_budget}) is greater than max value \
             ({MAX_COMPUTE_UNIT_LIMIT})",
        ));
    }

    Ok(())
}

pub(crate) fn generator<CreateMessage>(
    accounts: Arc<AccountsFile>,
    num_workers: usize,
    config: AttackProgramConfig,
    are_writable_accounts: bool,
    create_message: CreateMessage,
) -> impl FnMut(&Bank) -> (Vec<SanitizedTransaction>, usize) + Send
where
    CreateMessage: Fn(&Keypair, &Pubkey, &[AccountMeta], u32) -> Message + Send,
{
    // To enforce each transaction within the batch to be paid by a new payer
    // This is to reduce AccountsInUse errors
    let num_payers = accounts.payers.len();
    if num_payers < config.transaction_batch_size * num_workers {
        warn!(
            "Number of payers ({} is less than number of workers by batch size ({} x {}).This \
             will lead to AccountInUse errors.",
            config.transaction_batch_size, num_payers, num_workers
        );
    }
    let program_id = accounts
        .owner_program_id
        .expect("`owner_program_id` presence is checked during the config validation");
    let num_max_accounts = accounts.max_size.len();
    let accounts_meta: Vec<AccountMeta> = accounts
        .max_size
        .iter()
        .map(|account| AccountMeta {
            pubkey: account.pubkey(),
            is_signer: false,
            is_writable: are_writable_accounts,
        })
        .collect();

    let accounts_batch_size = config.transaction_batch_size * config.num_accounts_per_tx;

    let num_batches = num_max_accounts / accounts_batch_size;
    let mut batch_index = 0;

    // We want to use a new payer for each run of the closure.
    // It is impossible to use cyclic iterator because it would reference payers.
    let mut payer_index = 0;
    move |bank: &Bank| {
        let blockhash = bank.last_blockhash();

        // Splits all the accounts into set of batches, which are evenly distributed among workers.
        let worker_index = batch_index % num_workers;

        let accounts_batch = {
            let begin = batch_index * accounts_batch_size;
            let end = begin + accounts_batch_size;
            &accounts_meta[begin..end]
        };

        let transactions = accounts_batch
            .chunks(config.num_accounts_per_tx)
            .map(|tx_accounts| {
                let payer = &accounts.payers[payer_index];
                payer_index = (payer_index + 1) % num_payers;

                let message = create_message(
                    payer,
                    &program_id,
                    tx_accounts,
                    config.transaction_cu_budget,
                );
                Transaction::new(&[payer], message, blockhash)
            })
            .map(SanitizedTransaction::from_transaction_for_tests)
            .collect::<Vec<_>>();

        batch_index = (batch_index + 1) % num_batches;
        (transactions, worker_index)
    }
}
