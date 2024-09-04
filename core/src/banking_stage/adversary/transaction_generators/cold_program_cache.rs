//! Creates a generator that executes a program from a list of programs
use {
    super::TransactionGenerator,
    crate::banking_stage::{
        adversary::generator_templates::replay_attack_program::verify_common, BankingStage,
    },
    block_generator_stress_test::BlockGeneratorStressTestInstruction,
    solana_adversary::{
        accounts_file::AccountsFile,
        adversary_feature_set::replay_stage_attack::{Attack, AttackProgramConfig},
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

pub fn verify(accounts: &AccountsFile, attack: &Attack) -> Result<(), String> {
    let Attack::ColdProgramCache(attack) = attack else {
        panic!("Unexpected Attack passed into `_program::verify`: {attack:?}",);
    };

    let AttackProgramConfig {
        transaction_batch_size,
        transaction_cu_budget,
        ..
    } = attack;

    verify_common(
        *transaction_batch_size,
        *transaction_cu_budget,
        accounts.payers.len(),
    )?;

    let num_workers = BankingStage::default_num_workers().get();
    let max_num_independent_transactions = transaction_batch_size * num_workers;
    if accounts.program_ids_jit_attack.len() < max_num_independent_transactions {
        return Err(format!(
            "Not enough \"program_ids\" accounts: need at least \
             {max_num_independent_transactions}to ensure each transaction is invoking a unique \
             program."
        ));
    }

    Ok(())
}

fn create_nop_instruction(payer: &Keypair, program_id: Pubkey) -> Instruction {
    let random_data = vec![];
    let data = BlockGeneratorStressTestInstruction::Nop { random_data };
    let accounts_meta = vec![AccountMeta {
        pubkey: payer.pubkey(),
        is_signer: true,
        is_writable: false,
    }];
    Instruction::new_with_borsh(program_id, &data, accounts_meta)
}

pub(super) fn generator(
    accounts: Arc<AccountsFile>,
    num_workers: usize,
    config: AttackProgramConfig,
) -> TransactionGenerator {
    let AttackProgramConfig {
        transaction_batch_size,
        transaction_cu_budget,
        ..
    } = config;
    let num_payers = accounts.payers.len();

    let programs_batch_size = transaction_batch_size;
    let num_batches = accounts.program_ids_jit_attack.len() / programs_batch_size;
    let mut batch_index = 0;

    // We want to use a new payer for each run of the closure.
    // It is impossible to use cyclic iterator because it would reference payers.
    let mut payer_index = 0;
    Box::new(move |bank: &Bank| {
        let blockhash = bank.last_blockhash();
        let program_ids = &accounts.program_ids_jit_attack;

        // Splits all the accounts into set of batches, which are evenly distributed among workers.
        let worker_index = batch_index % num_workers;

        let programs_batch = {
            let begin = batch_index * programs_batch_size;
            let end = begin + programs_batch_size;
            &program_ids[begin..end]
        };

        let transactions = programs_batch
            .iter()
            .map(|program_id| {
                let payer = &accounts.payers[payer_index];
                payer_index = (payer_index + 1) % num_payers;

                let set_cu_instruction =
                    ComputeBudgetInstruction::set_compute_unit_limit(transaction_cu_budget);
                let message = Message::new(
                    &[
                        set_cu_instruction,
                        create_nop_instruction(payer, *program_id),
                    ],
                    Some(&payer.pubkey()),
                );

                Transaction::new(&[payer], message, blockhash)
            })
            .map(SanitizedTransaction::from_transaction_for_tests)
            .collect::<Vec<_>>();

        batch_index = (batch_index + 1) % num_batches;
        (transactions, worker_index)
    })
}
