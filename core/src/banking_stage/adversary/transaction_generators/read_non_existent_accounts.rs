//! Generates transactions with non-existent transactions accounts and valid payer accounts.

use {
    super::TransactionGenerator,
    crate::banking_stage::{
        adversary::{
            generator_components::IndexByModulo,
            generator_templates::max_accounts_tx::TX_MAX_ATTACK_ACCOUNTS_IN_PACKET,
        },
        consumer::TARGET_NUM_TRANSACTIONS_PER_BATCH,
    },
    rand::{seq::IteratorRandom, thread_rng},
    rayon::{
        iter::{IntoParallelIterator, ParallelIterator},
        ThreadPool,
    },
    solana_adversary::accounts_file::AccountsFile,
    solana_message::legacy::Message as LegacyMessage,
    solana_pubkey::{Pubkey, PUBKEY_BYTES},
    solana_runtime::bank::Bank,
    solana_signer::Signer,
    solana_transaction::{sanitized::SanitizedTransaction, Transaction},
    std::{iter, sync::Arc},
};

const NUM_NON_EXISTENT_ACCOUNTS_PER_TX: usize = TX_MAX_ATTACK_ACCOUNTS_IN_PACKET;

pub fn verify(accounts: &AccountsFile) -> Result<(), String> {
    if accounts.payers.len() < TARGET_NUM_TRANSACTIONS_PER_BATCH {
        return Err(format!(
            "Not enough `payer` accounts: need at least {TARGET_NUM_TRANSACTIONS_PER_BATCH}",
        ));
    }

    Ok(())
}

/// Creates transactions that use many non-existent accounts for input.
pub(super) fn generator(
    accounts: Arc<AccountsFile>,
    num_workers: usize,
    tx_generator_thread_pool: Arc<ThreadPool>,
) -> TransactionGenerator {
    // Used to distribute transactions across banking stage consume workers.
    let mut worker_index = IndexByModulo::new(num_workers);

    Box::new(move |bank: &Bank| {
        let blockhash = bank.last_blockhash();

        tx_generator_thread_pool.install(|| {
            let transactions: Vec<SanitizedTransaction> = accounts
                .payers
                .iter()
                .choose_multiple(&mut thread_rng(), TARGET_NUM_TRANSACTIONS_PER_BATCH)
                .into_par_iter()
                .map_init(thread_rng, |_rng, payer| {
                    // Create a transaction with a valid/funded payer account and non-existent input accounts.

                    // The base pubkey's bytes are randomly generated once and used to create other
                    // pubkeys. This is done to speed up the tx creation while still having a unique
                    // set of keys per transaction. we can do this because the test doesn't need
                    // the corresponding private keys.
                    let base_pubkey_bytes = rand::random::<[u8; PUBKEY_BYTES]>();

                    // As we are not running any instructions in this transaction, we do not really need
                    // to provide signatures for our testing accounts. The runtime will still need to
                    // load them, even if they did not sign.
                    let pubkeys: Vec<_> = iter::once(payer.pubkey())
                        .chain((0..NUM_NON_EXISTENT_ACCOUNTS_PER_TX).map(|i| {
                            let mut new_pubkey_bytes = base_pubkey_bytes;
                            new_pubkey_bytes[0] = new_pubkey_bytes[0].wrapping_add(i as u8);
                            Pubkey::new_from_array(new_pubkey_bytes)
                        }))
                        .collect::<Vec<_>>();

                    let message = LegacyMessage::new_with_compiled_instructions(
                        // Payer will sign and pay, so it is the only writable signer.
                        1,
                        0,
                        // All the other accounts are read-only accounts that did not provide a signature.
                        pubkeys
                            .len()
                            .saturating_sub(1)
                            .try_into()
                            .expect("`pubkeys.len()` fits into u8"),
                        pubkeys,
                        blockhash,
                        vec![],
                    );
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
        super::generator,
        crate::banking_stage::{
            adversary::{
                generator_templates::max_accounts_tx::TX_MAX_ATTACK_ACCOUNTS_IN_PACKET,
                test_helpers::{create_test_bank, setup_accounts, setup_test, TestAccounts},
            },
            consumer::TARGET_NUM_TRANSACTIONS_PER_BATCH,
        },
        jsonrpc_core::types::error::ErrorCode,
        rayon::ThreadPoolBuilder,
        serde_json::{json, Value},
        serial_test::serial,
        solana_adversary::{
            accounts_file::AccountsFile,
            adversary_feature_set::replay_stage_attack::{get_config, AdversarialConfig, Attack},
        },
        solana_message::MessageHeader,
        solana_pubkey::Pubkey,
        solana_rpc::{
            rpc::test_helpers::{parse_failure_response, parse_success_result},
            rpc_adversary::test_helpers::send_signed_request_sync,
        },
        solana_signer::Signer,
        std::sync::Arc,
    };

    const NUM_NON_EXISTENT_ACCOUNTS_PER_TX: u8 = TX_MAX_ATTACK_ACCOUNTS_IN_PACKET as u8;

    #[test]
    fn generate_one() {
        let accounts: AccountsFile = TestAccounts::new(TARGET_NUM_TRANSACTIONS_PER_BATCH, 0).into();
        let accounts_pubkeys: Vec<Pubkey> = accounts.payers.iter().map(|kp| kp.pubkey()).collect();
        let tx_generator_thread_pool =
            Arc::new(ThreadPoolBuilder::new().num_threads(2).build().unwrap());

        let mut tx_generator = generator(Arc::new(accounts), 1, tx_generator_thread_pool);

        let bank = create_test_bank();

        let (txs, worker_index) = tx_generator(&bank);
        assert_eq!(txs.len(), TARGET_NUM_TRANSACTIONS_PER_BATCH);
        assert_eq!(worker_index, 0);

        let tx = &txs[0];
        assert_eq!(tx.signatures().len(), 1);
        let message = tx.message();
        assert!(accounts_pubkeys.contains(message.fee_payer()));
        assert_eq!(
            message.header(),
            &MessageHeader {
                num_required_signatures: 1,
                num_readonly_signed_accounts: 0,
                num_readonly_unsigned_accounts: NUM_NON_EXISTENT_ACCOUNTS_PER_TX,
            }
        );
        assert_eq!(message.instructions(), &vec![]);
    }

    // TODO `[serial]` is necessary as the RPC configuration is a global singleton.  It would be
    // nice to move a to a more composable architecture and remove `[serial]`.
    #[test]
    #[serial]
    fn rpc_config_not_enough_payer_accounts() {
        let (mut meta, keypair, io, token) = setup_test();

        setup_accounts(&mut meta, 1, 0);

        let config = AdversarialConfig {
            selected_attack: Some(Attack::ReadNonExistentAccounts),
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
            "Not enough `payer` accounts: need at least 64".into(),
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

        setup_accounts(&mut meta, TARGET_NUM_TRANSACTIONS_PER_BATCH, 0);

        let config = AdversarialConfig {
            selected_attack: Some(Attack::ReadNonExistentAccounts),
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
