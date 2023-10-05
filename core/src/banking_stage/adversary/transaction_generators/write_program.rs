//! Creates a generator that executes the program, which writes into an account.

use {
    super::TransactionGenerator,
    block_generator_stress_test::BlockGeneratorStressTestInstruction,
    rand::{thread_rng, Rng},
    solana_adversary::{
        accounts_file::AccountsFile, adversary_feature_set::replay_stage_attack::WriteProgramConfig,
    },
    solana_compute_budget_interface::ComputeBudgetInstruction,
    solana_instruction::{AccountMeta, Instruction},
    solana_keypair::Keypair,
    solana_message::Message,
    solana_pubkey::Pubkey,
    solana_runtime::bank::Bank,
    solana_signer::Signer,
    solana_transaction::{sanitized::SanitizedTransaction, Transaction},
    std::sync::Arc,
};

pub(super) fn generator(
    accounts: Arc<AccountsFile>,
    num_workers: usize,
    config: WriteProgramConfig,
) -> TransactionGenerator {
    // To enforce each transaction within the batch to be paid by a new payer
    // This is to reduce AccountsInUse errors
    let num_payers = accounts.payers.len();
    if num_payers < config.transaction_batch_size * num_workers {
        warn!(
            "Number of payers ({} is less than number of workers by batch size ({} x {}). This \
             will lead to AccountInUse errors.",
            config.transaction_batch_size, num_payers, num_workers
        );
    }
    let program_id = accounts
        .owner_program_id
        .expect("`owner_program_id` presense is checked during the config validation");
    let num_max_accounts = accounts.max_size.len();
    let accounts_meta: Vec<AccountMeta> = accounts
        .max_size
        .iter()
        .map(|account| AccountMeta::new(account.pubkey(), false))
        .collect();

    let accounts_batch_size = config.transaction_batch_size * config.num_accounts_per_tx;

    let num_batches = num_max_accounts / accounts_batch_size;
    let mut batch_index = 0;

    // We want to use a new payer for each run of the closure.
    // It is impossible to use cyclic iterator because it would reference payers.
    let mut payer_index = 0;
    Box::new(move |bank: &Bank| {
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

                let message = create_write_message(
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
    })
}

fn create_write_message(
    payer: &Keypair,
    program_id: &Pubkey,
    accounts_meta: &[AccountMeta],
    transaction_cu_budget: u32,
) -> Message {
    let rnd = thread_rng().gen_range(0..=u64::MAX);
    let data = BlockGeneratorStressTestInstruction::WriteAccounts {
        value: 128,
        random: rnd,
    };

    // Explicitly specify the CU budget to avoid dropping some transactions on the CU check side.
    // The constraint is that 48M/64 > CU limit.
    // To maximize number of txs in the block, it is beneficial to set CU limit to be close to the
    // real CU consumption.
    // Default is 200k.
    let set_cu_instruction =
        ComputeBudgetInstruction::set_compute_unit_limit(transaction_cu_budget);

    let write_instruction = Instruction::new_with_borsh(*program_id, &data, accounts_meta.to_vec());
    Message::new(
        &[set_cu_instruction, write_instruction],
        Some(&payer.pubkey()),
    )
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        solana_ledger::genesis_utils::GenesisConfigInfo,
        solana_runtime::{bank::Bank, genesis_utils::create_genesis_config},
    };

    fn create_test_bank() -> Arc<Bank> {
        let GenesisConfigInfo { genesis_config, .. } = create_genesis_config(10_000);
        Bank::new_no_wallclock_throttle_for_tests(&genesis_config).0
    }

    #[test]
    fn test_generator_write_program() {
        let num_workers = 1;
        let num_payers = 64;
        let num_max_sized_accounts = 1024;
        let owner_program_id = Pubkey::default();
        let payers_accounts: Vec<Keypair> = (0..num_payers).map(|_| Keypair::new()).collect();
        let max_size_accounts: Vec<Keypair> = (0..num_max_sized_accounts)
            .map(|_| Keypair::new())
            .collect();
        let accounts = Arc::new(AccountsFile::with_payers_and_max_size(
            &owner_program_id,
            &payers_accounts,
            &max_size_accounts,
        ));
        let config = WriteProgramConfig {
            transaction_batch_size: 32,
            num_accounts_per_tx: 8,
            transaction_cu_budget: 5000,
            use_failed_transaction_hotpath: false,
        };
        let mut tx_generator = generator(accounts, num_workers, config.clone());

        let bank = create_test_bank();

        let (txs, _worker_id) = tx_generator(&bank);

        assert_eq!(txs.len(), config.transaction_batch_size);

        // In addition to the accounts to be modified, tx also includes a payer, the owner program
        // id, and the compute budget program id.
        let expected_num_accounts_per_tx = config.num_accounts_per_tx + 3;
        for tx in txs {
            let message = tx.message();

            assert_eq!(message.account_keys().len(), expected_num_accounts_per_tx);
            let instructions = &message.instructions();
            assert_eq!(instructions.len(), 2);

            let mut ix_iter = message.program_instructions_iter();
            ix_iter.next();
            assert_eq!(
                ix_iter.next().map(|(program_id, _ix)| program_id),
                Some(&owner_program_id)
            );
        }
    }
}
