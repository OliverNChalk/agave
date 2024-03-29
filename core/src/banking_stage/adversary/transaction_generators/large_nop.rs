//! Creates a generator that spams large transactions with low CUs.

use {
    super::TransactionGenerator,
    crate::banking_stage::adversary::generator_templates::replay_attack_program,
    block_generator_stress_test::{BlockGeneratorStressTestInstruction, LARGE_NOP_DATA_SIZE},
    rand::{seq::SliceRandom, thread_rng, Rng},
    solana_adversary::{
        accounts_file::AccountsFile,
        adversary_feature_set::replay_stage_attack::{Attack, LargeNopAttackConfig},
    },
    solana_compute_budget_interface::ComputeBudgetInstruction,
    solana_instruction::Instruction,
    solana_keypair::Keypair,
    solana_message::Message,
    solana_pubkey::Pubkey,
    solana_runtime::bank::Bank,
    solana_signer::Signer,
    solana_transaction::{sanitized::SanitizedTransaction, Transaction},
    std::sync::Arc,
};

pub fn verify(accounts: &AccountsFile, attack: &Attack) -> Result<(), String> {
    let Attack::LargeNop(attack_config) = attack else {
        panic!("Unexpected Attack passed into `large_nop::verify`: {attack:?}",);
    };

    if accounts.owner_program_id.is_none() {
        return Err("`owner_program_id` must be set.".to_string());
    }

    // Data size above this will cause transactions to fail.
    if attack_config.tx_data_size > LARGE_NOP_DATA_SIZE {
        return Err(format!(
            "`tx_data_size` ({}) must be less than or equal to {}",
            attack_config.tx_data_size, LARGE_NOP_DATA_SIZE
        ));
    }

    replay_attack_program::verify_common(
        attack_config.common.transaction_batch_size,
        attack_config.common.transaction_cu_budget,
        accounts.payers.len(),
    )
}

pub(super) fn generator(
    accounts: Arc<AccountsFile>,
    num_workers: usize,
    config: LargeNopAttackConfig,
) -> TransactionGenerator {
    let mut worker_index = 0;
    let program_id = accounts
        .owner_program_id
        .expect("`owner_program_id` presence is checked during the config validation");
    Box::new(move |bank: &Bank| {
        let blockhash = bank.last_blockhash();

        let transactions = accounts
            .payers
            .choose_multiple(&mut thread_rng(), config.common.transaction_batch_size)
            .map(|payer| {
                let message = create_nop_message(
                    payer,
                    &program_id,
                    config.common.transaction_cu_budget,
                    config.tx_data_size,
                );
                Transaction::new(&[payer], message, blockhash)
            })
            .map(SanitizedTransaction::from_transaction_for_tests)
            .collect::<Vec<_>>();

        let current_worker_index = worker_index;
        worker_index = (worker_index + 1) % num_workers;
        (transactions, current_worker_index)
    })
}

fn create_nop_message(
    payer: &Keypair,
    program_id: &Pubkey,
    transaction_cu_budget: u32,
    tx_data_size: usize,
) -> Message {
    let rnd = thread_rng().gen_range(0..=u64::MAX);
    let random_data: Vec<u8> = rnd
        .to_le_bytes()
        .iter()
        .cycle()
        .take(tx_data_size)
        .cloned()
        .collect();
    let data = BlockGeneratorStressTestInstruction::Nop {
        random_data: random_data.into_boxed_slice(),
    };

    // Explicitly specify the CU budget to avoid dropping some transactions on the CU check side.
    // The constraint is that 48M/64 > CU limit.
    // To maximize number of txs in the block, it is beneficial to set CU limit to be close to the
    // real CU consumption.
    // Default is 200k.
    let set_cu_instruction =
        ComputeBudgetInstruction::set_compute_unit_limit(transaction_cu_budget);

    let nop_instruction = Instruction::new_with_borsh(*program_id, &data, vec![]);
    Message::new(
        &[set_cu_instruction, nop_instruction],
        Some(&payer.pubkey()),
    )
}

#[cfg(test)]
mod tests {
    use {
        crate::banking_stage::adversary::test_helpers::{setup_accounts, setup_test},
        jsonrpc_core::{types::error::ErrorCode, Value},
        serde_json::json,
        serial_test::serial,
        solana_adversary::adversary_feature_set::replay_stage_attack::{
            get_config, AdversarialConfig, Attack, AttackProgramConfig, LargeNopAttackConfig,
        },
        solana_rpc::{
            rpc::test_helpers::{parse_failure_response, parse_success_result},
            rpc_adversary::test_helpers::send_signed_request_sync,
        },
    };

    // TODO `[serial]` is necessary as the RPC configuration is a global singleton.  It would be
    // nice to move a to a more composable architecture and remove `[serial]`.
    #[test]
    #[serial]
    fn rpc_config_invalid() {
        let (mut meta, keypair, io, token) = setup_test();

        setup_accounts(&mut meta, 128, 32);

        let config = AdversarialConfig {
            selected_attack: Some(Attack::LargeNop(LargeNopAttackConfig {
                common: AttackProgramConfig {
                    transaction_batch_size: 0,
                    num_accounts_per_tx: 8,
                    transaction_cu_budget: 10000,
                    use_failed_transaction_hotpath: false,
                },
                tx_data_size: 100,
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
            selected_attack: Some(Attack::LargeNop(LargeNopAttackConfig {
                common: AttackProgramConfig {
                    transaction_batch_size: 4,
                    num_accounts_per_tx: 8,
                    transaction_cu_budget: 10000,
                    use_failed_transaction_hotpath: false,
                },
                tx_data_size: 100,
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
            selected_attack: Some(Attack::LargeNop(LargeNopAttackConfig {
                common: AttackProgramConfig {
                    transaction_batch_size: 4,
                    num_accounts_per_tx: 8,
                    transaction_cu_budget: 10000,
                    use_failed_transaction_hotpath: false,
                },
                tx_data_size: 100,
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
