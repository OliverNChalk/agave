//! Creates a generator that spams large transactions with low CUs.

use {
    super::TransactionGenerator,
    crate::banking_stage::adversary::{
        generator_components::IndexByModulo, generator_templates::replay_attack_program,
    },
    block_generator_stress_test::{BlockGeneratorStressTestInstruction, LARGE_NOP_DATA_SIZE},
    rand::{seq::SliceRandom, thread_rng, Rng},
    rayon::iter::{IntoParallelIterator, ParallelIterator},
    solana_adversary::{
        accounts_file::AccountsFile,
        adversary_feature_set::replay_stage_attack::{Attack, LargeNopAttackConfig},
    },
    solana_compute_budget_interface::ComputeBudgetInstruction,
    solana_instruction::Instruction,
    solana_message::Message,
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

fn create_nop_instruction(accounts: Arc<AccountsFile>, tx_data_size: usize) -> Instruction {
    let program_id = accounts
        .owner_program_id
        .expect("`owner_program_id` presence is checked during the config validation");
    let rnd = rand::thread_rng().gen::<u64>();
    let random_data: Vec<u8> = std::iter::repeat_with(|| rnd.to_le_bytes().to_vec())
        .flatten()
        .take(tx_data_size)
        .collect();
    let data = BlockGeneratorStressTestInstruction::Nop {
        random_data: random_data.into_boxed_slice(),
    };
    Instruction::new_with_borsh(program_id, &data, vec![])
}

pub(super) fn generator(
    accounts: Arc<AccountsFile>,
    num_workers: usize,
    config: LargeNopAttackConfig,
) -> TransactionGenerator {
    // Used to distribute transactions across banking stage consume workers.
    let mut worker_index = IndexByModulo::new(num_workers);

    // Transaction generation is the bottleneck, so parallelize it. Currently
    // limiting to just 2 threads because we can exceed the shred limit when
    // moving to 4 threads and beyond.
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();

    // Explicitly specify the CU budget to avoid dropping some transactions on the CU check side.
    // The constraint is that 48M/64 > CU limit.
    // To maximize number of txs in the block, it is beneficial to set CU limit to be close to the
    // real CU consumption.
    // Default is 200k.
    let set_cu_instruction =
        ComputeBudgetInstruction::set_compute_unit_limit(config.common.transaction_cu_budget);

    let nop_instruction = create_nop_instruction(accounts.clone(), config.tx_data_size);

    Box::new(move |bank: &Bank| {
        let blockhash = bank.last_blockhash();
        let instructions = &[set_cu_instruction.clone(), nop_instruction.clone()];
        thread_pool.install(|| {
            let transactions: Vec<SanitizedTransaction> = (0..config.common.transaction_batch_size)
                .into_par_iter()
                .map_init(thread_rng, |rng, _| {
                    // Unique payers for entropy in the transaction generation.
                    let payer = accounts.payers.choose(rng).expect("should have payers");
                    let message = Message::new(instructions, Some(&payer.pubkey()));
                    let transaction = Transaction::new(&[payer], message, blockhash);
                    SanitizedTransaction::from_transaction_for_tests(transaction)
                })
                .collect();
            (transactions, worker_index.next())
        })
    })
}

#[cfg(test)]
mod tests {
    use {
        crate::banking_stage::adversary::test_helpers::{
            create_test_bank, setup_accounts, setup_test, TestAccounts,
        },
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
        std::sync::Arc,
    };

    #[test]
    fn test_generator_produces_transactions() {
        // Prepare all necessary inputs to create and call tx generator.
        let accounts = Arc::new(TestAccounts::new(10_000, 0).into());
        let num_workers = 4;
        let config = LargeNopAttackConfig::default();
        let bank = create_test_bank();

        // Create the generator.
        let mut generator = super::generator(accounts, num_workers, config);

        // Generate a batch of transactions.
        let (transactions, _worker_index) = generator(&bank);

        // Verify transactions were generated.
        assert!(
            transactions.len() == config.common.transaction_batch_size,
            "Expected a full batch of tx to be generated",
        );
    }

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
