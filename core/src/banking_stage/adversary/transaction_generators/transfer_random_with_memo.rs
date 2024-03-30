//! Generates transfers between a set of accounts with a large memo to create transactions with the maximum size.

use {
    super::TransactionGenerator,
    crate::banking_stage::consumer::TARGET_NUM_TRANSACTIONS_PER_BATCH,
    rand::{seq::SliceRandom, thread_rng},
    solana_adversary::accounts_file::AccountsFile,
    solana_runtime::bank::Bank,
    solana_signer::Signer,
    solana_system_interface::instruction as system_instruction,
    solana_transaction::{sanitized::SanitizedTransaction, Transaction},
    spl_memo_interface::instruction::build_memo,
    std::sync::Arc,
};

const BATCH_SIZE: usize = TARGET_NUM_TRANSACTIONS_PER_BATCH;

pub fn verify(accounts: &AccountsFile) -> Result<(), String> {
    if accounts.payers.len() < BATCH_SIZE {
        return Err(format!(
            "Not enough `payer` accounts: need at least {BATCH_SIZE}",
        ));
    }

    Ok(())
}

pub(super) fn generator(accounts: Arc<AccountsFile>, num_workers: usize) -> TransactionGenerator {
    let mut worker_index = 0;
    // This memo was carefully created to pad out the data section of the tx to
    // force it to max size (currently 1232 bytes).
    let memo = "The quick brown fox jumped over the lazy dog. ".repeat(22) + "!";
    let memo_instruction = build_memo(&spl_memo_interface::v3::id(), memo.as_bytes(), &[]);
    Box::new(move |bank: &Bank| {
        let blockhash = bank.last_blockhash();

        let transactions = accounts
            .payers
            .choose_multiple(&mut thread_rng(), BATCH_SIZE)
            .map(|account| {
                // Just transfer to self for 0 lamports to avoid actually
                // modifying accounts and to keep CUs lower.
                let transfer_instruction =
                    system_instruction::transfer(&account.pubkey(), &account.pubkey(), 0);

                // Combine instructions into a single transaction.
                Transaction::new_signed_with_payer(
                    &[transfer_instruction, memo_instruction.clone()],
                    Some(&account.pubkey()),
                    &[account],
                    blockhash,
                )
            })
            .map(SanitizedTransaction::from_transaction_for_tests)
            .collect::<Vec<_>>();

        let current_worker_index = worker_index;
        worker_index = (worker_index + 1) % num_workers;
        (transactions, current_worker_index)
    })
}

#[cfg(test)]
pub mod tests {
    use {
        crate::banking_stage::adversary::test_helpers::{setup_accounts, setup_test},
        jsonrpc_core::{types::error::ErrorCode, Value},
        serde_json::json,
        serial_test::serial,
        solana_adversary::adversary_feature_set::replay_stage_attack::{
            get_config, AdversarialConfig, Attack,
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
    fn rpc_config_no_accounts() {
        let (meta, keypair, io, token) = setup_test();

        // Set invalid attack: no accounts have been specified
        let config = AdversarialConfig {
            selected_attack: Some(Attack::TransferRandomWithMemo),
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
            "Cannot launch attack: accounts were not setup".into(),
        );
        assert_eq!(result, expected);
        assert_eq!(
            AdversarialConfig::default(),
            get_config(),
            "Invalid config update should not change the config"
        );
    }

    // TODO See `rpc_config_no_accounts()` for why `[serial]` is necessary.
    #[test]
    #[serial]
    fn rpc_config_valid() {
        let (mut meta, keypair, io, token) = setup_test();

        setup_accounts(&mut meta, 128, 32);

        let config = AdversarialConfig {
            selected_attack: Some(Attack::TransferRandomWithMemo),
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
