//! Creates a generator that executes the program, which calls recursively read function

use {
    super::TransactionGenerator,
    crate::banking_stage::adversary::generator_templates::replay_attack_program,
    block_generator_stress_test::BlockGeneratorStressTestInstruction,
    rand::{thread_rng, RngCore},
    solana_adversary::{
        accounts_file::AccountsFile,
        adversary_feature_set::replay_stage_attack::{Attack, AttackProgramConfig},
    },
    solana_compute_budget_interface::ComputeBudgetInstruction,
    solana_instruction::{AccountMeta, Instruction},
    solana_keypair::Keypair,
    solana_message::Message,
    solana_pubkey::Pubkey,
    solana_signer::Signer,
    std::sync::Arc,
};

/// Empirically found that higher values lead to error
/// "Cross-program invocation call depth too deep"
const MAX_CPI_DEPTH: u8 = 4;

pub fn verify(accounts: &AccountsFile, attack: &Attack) -> Result<(), String> {
    let Attack::RecursiveProgram(attack) = attack else {
        panic!("Unexpected Attack passed into `recursive_program::verify`: {attack:?}",);
    };
    if accounts.owner_program_id.is_none() {
        return Err("`owner_program_id` must be set.".to_string());
    }
    replay_attack_program::verify_replay_program_execution_attack(accounts, *attack)
}

pub(super) fn generator(
    accounts: Arc<AccountsFile>,
    num_workers: usize,
    config: AttackProgramConfig,
) -> TransactionGenerator {
    Box::new(replay_attack_program::generator(
        accounts,
        num_workers,
        config,
        true, // TODO(klykov) check if matters
        create_recursive_message,
    ))
}

fn create_recursive_message(
    payer: &Keypair,
    program_id: &Pubkey,
    accounts_meta: &[AccountMeta],
    transaction_cu_budget: u32,
) -> Message {
    let data = BlockGeneratorStressTestInstruction::Recurse {
        depth: MAX_CPI_DEPTH,
        random: thread_rng().next_u64(),
    };

    // Explicitly specify the CU budget to avoid dropping some transactions on the CU check side.
    // The constraint is that 48M/64 > CU limit.
    // To maximize number of txs in the block, it is beneficial to set CU limit to be close to the
    // real CU consumption.
    // Default is 200k.
    let set_cu_instruction =
        ComputeBudgetInstruction::set_compute_unit_limit(transaction_cu_budget);

    // For the CPI, we need to pass program_id as well
    let program_account_meta = AccountMeta {
        pubkey: *program_id,
        is_signer: false,
        is_writable: false,
    };
    let accounts_meta = [&[program_account_meta], accounts_meta].concat();

    let recursive_instruction = Instruction::new_with_borsh(*program_id, &data, accounts_meta);
    Message::new(
        &[set_cu_instruction, recursive_instruction],
        Some(&payer.pubkey()),
    )
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        crate::banking_stage::adversary::test_helpers::{
            create_test_bank, setup_accounts, setup_test,
        },
        jsonrpc_core::Value,
        serde_json::json,
        serial_test::serial,
        solana_adversary::{
            accounts_file::AccountsFile,
            adversary_feature_set::replay_stage_attack::{
                get_config, AdversarialConfig, Attack, AttackProgramConfig,
            },
        },
        solana_rpc::{
            rpc::test_helpers::parse_success_result,
            rpc_adversary::test_helpers::send_signed_request_sync,
        },
    };

    #[test]
    fn test_generator_recursive_program() {
        let num_workers = 1;
        let num_payers = 64;
        let num_max_sized_accounts = 1024;
        let owner_program_id = Keypair::new().pubkey();
        let payers_accounts: Vec<Keypair> = (0..num_payers).map(|_| Keypair::new()).collect();
        let max_size_accounts: Vec<Keypair> = (0..num_max_sized_accounts)
            .map(|_| Keypair::new())
            .collect();
        let accounts = Arc::new(AccountsFile::with_payers_and_max_size(
            &owner_program_id,
            &payers_accounts,
            &max_size_accounts,
        ));
        let config = AttackProgramConfig {
            transaction_batch_size: 32,
            num_accounts_per_tx: 8,
            transaction_cu_budget: 5000,
            use_failed_transaction_hotpath: false,
        };
        let mut tx_generator = generator(accounts, num_workers, config);

        let bank = create_test_bank();

        let (txs, _worker_id) = tx_generator(&bank);

        assert_eq!(txs.len(), config.transaction_batch_size);

        for tx in txs {
            let message = tx.message();

            // In addition to the accounts to be read, tx also includes a payer, the owner program
            // id, and the compute budget program id.
            let expected_num_accounts_per_tx = config.num_accounts_per_tx + 3;
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

    // TODO `[serial]` is necessary as the RPC configuration is a global singleton.  It would be
    // nice to move a to a more composable architecture and remove `[serial]`.
    #[test]
    #[serial]
    fn rpc_config_valid() {
        let (mut meta, keypair, io, token) = setup_test();

        setup_accounts(&mut meta, 128, 32);

        let config = AdversarialConfig {
            selected_attack: Some(Attack::RecursiveProgram(AttackProgramConfig {
                transaction_batch_size: 1,
                num_accounts_per_tx: 8,
                transaction_cu_budget: 10000,
                use_failed_transaction_hotpath: false,
            })),
        };
        let rsp = send_signed_request_sync(
            meta.clone(),
            &io,
            &keypair,
            "configureReplayStageAttack",
            &token,
            &config,
        );
        let result: Value = parse_success_result(rsp);
        assert_eq!(result, json!(null));
        assert_eq!(
            config,
            get_config(),
            "Config update must be reflected internally"
        );
    }
}
