//! Creates a generator that executes the program, which writes into an account.

use {
    super::TransactionGenerator,
    crate::banking_stage::adversary::generator_templates::replay_attack_program,
    block_generator_stress_test::BlockGeneratorStressTestInstruction,
    rand::{thread_rng, Rng},
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

pub fn verify(accounts: &AccountsFile, attack: &Attack) -> Result<(), String> {
    let Attack::WriteProgram(attack) = attack else {
        panic!("Unexpected Attack passed into `write_program::verify`: {attack:?}",);
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
        true,
        create_write_message,
    ))
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
        crate::banking_stage::adversary::test_helpers::{
            create_test_bank, setup_accounts, setup_test,
        },
        jsonrpc_core::{types::error::ErrorCode, Value},
        serde_json::json,
        serial_test::serial,
        solana_adversary::{
            accounts_file::AccountsFile,
            adversary_feature_set::replay_stage_attack::{
                get_config, AdversarialConfig, Attack, AttackProgramConfig,
            },
        },
        solana_rpc::{
            rpc::test_helpers::{parse_failure_response, parse_success_result},
            rpc_adversary::test_helpers::send_signed_request_sync,
        },
        std::sync::Arc,
    };

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
        let accounts = Arc::new(AccountsFile::new(
            Some(owner_program_id),
            Some(&payers_accounts),
            Some(&max_size_accounts),
            None,
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

    // TODO `[serial]` is necessary as the RPC configuration is a global singleton.  It would be
    // nice to move a to a more composable architecture and remove `[serial]`.
    #[test]
    #[serial]
    fn rpc_config_invalid() {
        let (mut meta, keypair, io, token) = setup_test();

        setup_accounts(&mut meta, 128, 32);

        let config = AdversarialConfig {
            selected_attack: Some(Attack::WriteProgram(AttackProgramConfig {
                transaction_batch_size: 0,
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
        let result = parse_failure_response(rsp);
        let expected = (
            ErrorCode::InvalidParams.code(),
            "`transaction_batch_size` (0) must be in range [1, 64]".into(),
        );
        assert_eq!(result, expected);
        assert_eq!(
            AdversarialConfig::default(),
            get_config(),
            "Invalid config update should not change the config"
        );
    }

    // TODO See `rpc_config_invalid()` for why `[serial]` is necessary.
    #[test]
    #[serial]
    fn rpc_config_not_enough_payer_accounts() {
        let (mut meta, keypair, io, token) = setup_test();

        setup_accounts(&mut meta, 2, 1);

        let config = AdversarialConfig {
            selected_attack: Some(Attack::WriteProgram(AttackProgramConfig {
                transaction_batch_size: 4,
                num_accounts_per_tx: 8,
                transaction_cu_budget: 10000,
                use_failed_transaction_hotpath: false,
            })),
        };
        let response = send_signed_request_sync(
            meta.clone(),
            &io,
            &keypair,
            "configureReplayStageAttack",
            &token,
            &config,
        );
        let result = parse_failure_response(response);
        let expected = (
            ErrorCode::InvalidParams.code(),
            "Not enough \"payer\" accounts: need at least 16\n\"payer\" accounts: 2 to avoid \
             having AccountsInUse errors"
                .into(),
        );
        assert_eq!(result, expected);
        assert_eq!(
            AdversarialConfig::default(),
            get_config(),
            "Invalid config update should not change the config"
        );
    }

    // TODO See `rpc_config_invalid()` for why `[serial]` is necessary.
    #[test]
    #[serial]
    fn rpc_config_valid() {
        let (mut meta, keypair, io, token) = setup_test();

        setup_accounts(&mut meta, 128, 32);

        let config = AdversarialConfig {
            selected_attack: Some(Attack::WriteProgram(AttackProgramConfig {
                transaction_batch_size: 4,
                num_accounts_per_tx: 8,
                transaction_cu_budget: 10000,
                use_failed_transaction_hotpath: false,
            })),
        };
        let response = send_signed_request_sync(
            meta.clone(),
            &io,
            &keypair,
            "configureReplayStageAttack",
            &token,
            &config,
        );
        let result: Value = parse_success_result(response);
        assert_eq!(result, json!(null));
        assert_eq!(
            config,
            get_config(),
            "Config update must be reflected internally"
        );
    }
}
