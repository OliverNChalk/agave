use {
    super::TransactionGenerator,
    crate::banking_stage::adversary::{
        generator_components::cycler, generator_templates::max_size_accounts_tx,
    },
    solana_adversary::accounts_file::AccountsFile,
    solana_keypair::Keypair,
    solana_message::legacy::Message as LegacyMessage,
    solana_runtime::bank::Bank,
    solana_signer::Signer,
    solana_transaction::Transaction,
    std::{iter, sync::Arc},
};

pub fn verify(accounts: &AccountsFile) -> Result<(), String> {
    max_size_accounts_tx::verify(accounts)
}

/// Creates transactions that use maximum number of maximum sized accounts for input.  Overloading
/// the replay machinery IO subsystem.
pub(super) fn generator(accounts: Arc<AccountsFile>, num_workers: usize) -> TransactionGenerator {
    Box::new(max_size_accounts_tx::generator(
        accounts,
        num_workers,
        move |bank: &Bank, payer: &Keypair, tx_accounts: cycler::ChunkIter<'_, '_, Keypair>| {
            let blockhash = bank.last_blockhash();

            // As we are not running any instructions in this transaction, we do not really need
            // to provide signatures for our testing accounts.  The runtime will still need to
            // load them, even if they did not sign.
            let pubkeys = iter::once(payer.pubkey())
                .chain(tx_accounts.map(Keypair::pubkey))
                .collect::<Vec<_>>();

            let message = LegacyMessage::new_with_compiled_instructions(
                // Payer will sign and pay, so it is the only writable signer.
                1,
                0,
                // No read-only accounts that did not provide a signature.  Meaning all the other
                // accounts are read/write accounts with no signature.
                0,
                pubkeys,
                blockhash,
                vec![],
            );
            Transaction::new(&[&payer], message, blockhash)
        },
    ))
}

#[cfg(test)]
mod tests {
    use {
        super::generator,
        crate::banking_stage::adversary::{
            generator_templates::{
                max_size_accounts_tx::TX_MAX_NUM_MAX_SIZE_ACCOUNTS, rotate_accounts::BATCH_SIZE,
            },
            test_helpers::{create_test_bank, setup_accounts, setup_test, TestAccounts},
        },
        jsonrpc_core::types::error::ErrorCode,
        serde_json::{json, Value},
        serial_test::serial,
        solana_adversary::{
            accounts_file::AccountsFile,
            adversary_feature_set::replay_stage_attack::{get_config, AdversarialConfig, Attack},
        },
        solana_keypair::Keypair,
        solana_message::MessageHeader,
        solana_rpc::{
            rpc::test_helpers::{parse_failure_response, parse_success_result},
            rpc_adversary::test_helpers::send_signed_request_sync,
        },
        solana_signer::Signer,
        std::{iter, sync::Arc},
    };

    #[test]
    fn generate_one() {
        let accounts: AccountsFile = TestAccounts::new(1, 64).into();
        let first_payer = accounts.payers[0].pubkey();
        let max_size_pubkeys = accounts
            .max_size
            .iter()
            .map(Keypair::pubkey)
            .collect::<Vec<_>>();

        let mut tx_generator = generator(Arc::new(accounts), 1);

        let bank = create_test_bank();

        let (txs, worker_index) = tx_generator(&bank);
        assert_eq!(txs.len(), BATCH_SIZE);
        assert_eq!(worker_index, 0);

        let tx = &txs[0];
        assert_eq!(tx.signatures().len(), 1);
        let message = tx.message();
        assert_eq!(message.fee_payer(), &first_payer);
        assert_eq!(
            message.header(),
            &MessageHeader {
                num_required_signatures: 1,
                num_readonly_signed_accounts: 0,
                num_readonly_unsigned_accounts: 0,
            }
        );
        assert_eq!(message.instructions(), &vec![]);
        assert_eq!(
            message.account_keys().iter().collect::<Vec<_>>(),
            iter::once(&first_payer)
                .chain(max_size_pubkeys[0..TX_MAX_NUM_MAX_SIZE_ACCOUNTS].iter())
                .collect::<Vec<_>>(),
        );
    }

    // TODO `[serial]` is necessary as the RPC configuration is a global singleton.  It would be
    // nice to move a to a more composable architecture and remove `[serial]`.
    #[test]
    #[serial]
    fn rpc_config_not_enough_payer_accounts() {
        let (mut meta, keypair, io, token) = setup_test();

        setup_accounts(&mut meta, 0, BATCH_SIZE * TX_MAX_NUM_MAX_SIZE_ACCOUNTS);

        let config = AdversarialConfig {
            selected_attack: Some(Attack::WriteMaxSizeAccounts),
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
            "Not enough `payer` accounts: need at least 1\n`payer` accounts: 0".into(),
        );
        assert_eq!(result, expected);
        assert_eq!(
            AdversarialConfig::default(),
            get_config(),
            "Invalid config update should not change the config"
        );
    }

    // TODO `[serial]` is necessary as the RPC configuration is a global singleton.  It would be
    // nice to move a to a more composable architecture and remove `[serial]`.
    #[test]
    #[serial]
    fn rpc_config_not_enough_max_size_accounts() {
        let (mut meta, keypair, io, token) = setup_test();

        setup_accounts(&mut meta, BATCH_SIZE, 1);

        let config = AdversarialConfig {
            selected_attack: Some(Attack::WriteMaxSizeAccounts),
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
            "Not enough `max_size` accounts: need at least 6\n`max_size` accounts: 1".into(),
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

        setup_accounts(
            &mut meta,
            BATCH_SIZE,
            TX_MAX_NUM_MAX_SIZE_ACCOUNTS * BATCH_SIZE,
        );

        let config = AdversarialConfig {
            selected_attack: Some(Attack::WriteMaxSizeAccounts),
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
