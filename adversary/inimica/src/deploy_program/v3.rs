//! Deploys a loader v3 program.

use {
    crate::{blockhash_cache::BlockhashCache, programs::ProgramElfRef},
    log::debug,
    solana_clock::Slot,
    solana_instruction::error::InstructionError,
    solana_keypair::Keypair,
    solana_loader_v3_interface::{
        instruction as loader_v3_instruction, state::UpgradeableLoaderState,
    },
    solana_pubkey::Pubkey,
    solana_rent::Rent,
    solana_rpc_client::nonblocking::rpc_client::RpcClient,
    solana_rpc_client_api::client_error::Error as RpcClientError,
    solana_signer::Signer,
    solana_transaction::Transaction,
    std::time::Duration,
    thiserror::Error,
    tokio::time::sleep,
};

#[derive(Error, Debug)]
pub enum DeployError {
    #[error("create_buffer() failed to create instructions: {0}")]
    CreateBufferInstructions(InstructionError),

    #[error("Buffer account creation tx failed: {0}")]
    BufferAccountCreationTx(RpcClientError),

    #[error("ELF size exceeded u32: {size}")]
    ElfSizeLimit { size: usize },

    #[error("Failed to write into program buffer at {buffer_address}, offset {offset}: {error}")]
    BufferWrite {
        buffer_address: Pubkey,
        offset: u32,
        error: RpcClientError,
    },

    #[error("deploy_with_max_program_len() failed to create instructions: {0}")]
    DeployWithMaxPrograLenInstructions(InstructionError),

    #[error("Failed to finalize the program deployment: {0}")]
    DeployWithMaxPrograLen(RpcClientError),

    #[error("Failed to retrieve signature status for the program deployment: {0}")]
    DeployWithMaxPrograLenSignature(RpcClientError),

    #[error("Failed to get an account balance for {address}: {error}")]
    GetBalance {
        address: Pubkey,
        error: RpcClientError,
    },

    #[error("Failed to delete a program buffer at {address}: {error}")]
    DeleteBuffer {
        address: Pubkey,
        error: RpcClientError,
    },

    #[error("Failed to delete a program account at {address}: {error}")]
    DeleteProgram {
        address: Pubkey,
        error: RpcClientError,
    },
}

/// Deploys the specified program at the specified address, recording the payer as the authority for
/// the program.  Returns the slot in which the program was deployed.
///
/// In case of any errors, deployed accounts are deleted and SOL locked in the rent for those
/// accounts is refunded to the payer, before the function returns.
///
/// If an error happens during the program finalization step, that is, the SBF code is incorrect,
/// the program account is not deleted.  While the program data account is deleted.  This is just an
/// implementation shortcoming as of right now.
pub async fn deploy(
    blockhash_cache: &BlockhashCache,
    rent: &Rent,
    client: &RpcClient,
    payer: Keypair,
    program_key: Keypair,
    elf: ProgramElfRef<'_>,
) -> Result<Slot, DeployError> {
    let buffer_key = create_buffer(blockhash_cache, rent, client, &payer, elf).await?;
    let buffer_address = buffer_key.pubkey();

    let res = fill_buffer(blockhash_cache, client, &payer, &buffer_address, elf).await;
    if let Err(err) = res {
        // Try to recovery some SOL in case of an error.
        let _ = delete_buffer(blockhash_cache, client, &payer, &buffer_address).await;
        return Err(err);
    }

    let res = finalize(
        blockhash_cache,
        rent,
        client,
        &payer,
        &program_key,
        &buffer_address,
        elf,
    )
    .await;
    match res {
        Ok(slot) => Ok(slot),
        Err(err) => {
            // Try to recovery some SOL in case of an error.
            let _ = delete_buffer(blockhash_cache, client, &payer, &buffer_address).await;
            Err(err)
        }
    }
}

async fn create_buffer(
    blockhash_cache: &BlockhashCache,
    rent: &Rent,
    client: &RpcClient,
    payer: &Keypair,
    elf: ProgramElfRef<'_>,
) -> Result<Keypair, DeployError> {
    let payer_address = &payer.pubkey();
    let buffer_key = Keypair::new();

    let program_len = elf.len();
    let space = UpgradeableLoaderState::size_of_buffer(program_len);
    let buffer_lamports = rent.minimum_balance(space);
    let instructions = loader_v3_instruction::create_buffer(
        payer_address,
        &buffer_key.pubkey(),
        payer_address,
        buffer_lamports,
        program_len,
    )
    .map_err(DeployError::CreateBufferInstructions)?;

    let transaction = Transaction::new_signed_with_payer(
        &instructions,
        Some(payer_address),
        &[payer, &buffer_key],
        blockhash_cache.get(),
    );
    let signature = client
        .send_and_confirm_transaction(&transaction)
        .await
        .map_err(DeployError::BufferAccountCreationTx)?;

    debug!(
        "Buffer {}: Created buffer for a program: {}",
        buffer_key.pubkey(),
        signature
    );

    Ok(buffer_key)
}

// TODO This function could be made more efficient, if we allow it to run more than one `write()`
// request in parallel.
//
// Transaction execution needs to happen in sequence, due to the write lock restriction on the
// target buffer.  But we do not need to wait for one write to be committed at any level before we
// run the next one.  Plus, a lot of work, such as network communication, could be performed in
// parallel regardless.
//
// When we are deploying multiple instances of the same program in parallel, we are getting
// parallelism at a higher level, so for an attack that deploys multiple copies it is not critical
// that this method does not have parallelism.
//
// Ideally, it would be nice if the method would take a number that would specify the maximum number
// of outstanding requests that could be in flight at any given point of time.  This would allow
// higher level operations to control the load they create.
//
// Another optimization point is to use `send_transaction_with_config()` instead of
// `send_and_confirm_transaction()`.  The latter, among other things, will do a "preflight check".
// There is little reason for simulating buffer write operations - we can just run them as is,
// potentially avoiding a delay.
async fn fill_buffer(
    blockhash_cache: &BlockhashCache,
    client: &RpcClient,
    payer: &Keypair,
    buffer_address: &Pubkey,
    elf: ProgramElfRef<'_>,
) -> Result<(), DeployError> {
    // TODO It would be nice to compute this based on the constants available in the SDK and other
    // crates.  For now the `simple_fill_buffer()` test makes sure that generated transactions are
    // within the size limit.  And this is the largest value for which the test still passes.
    const UPLOAD_CHUNK_SIZE: usize = 1008;

    let payer_address = &payer.pubkey();

    for (i, chunk) in elf.chunks(UPLOAD_CHUNK_SIZE).enumerate() {
        let offset = u32::try_from(i.saturating_mul(UPLOAD_CHUNK_SIZE))
            .map_err(|_err| DeployError::ElfSizeLimit { size: elf.len() })?;

        let instruction =
            loader_v3_instruction::write(buffer_address, payer_address, offset, chunk.to_vec());
        let transaction = Transaction::new_signed_with_payer(
            &[instruction],
            Some(payer_address),
            &[payer],
            blockhash_cache.get(),
        );

        let signature = client
            .send_and_confirm_transaction(&transaction)
            .await
            .map_err(|error| DeployError::BufferWrite {
                buffer_address: *buffer_address,
                offset,
                error,
            })?;

        debug!(
            "Buffer {}: Uploaded {} bytes into program buffer at {}: {}",
            buffer_address,
            chunk.len(),
            offset,
            signature
        );
    }

    Ok(())
}

// TODO invalidator: remove deprecated function and use loader-v4.
// `solana_loader_v3_interface::deploy_with_max_program_len`:
// Use loader-v4 instead.
// 'allow deprecated' added by fkouteib on 2/09/2025 upstream sync.
#[allow(deprecated)]
async fn finalize(
    blockhash_cache: &BlockhashCache,
    rent: &Rent,
    client: &RpcClient,
    payer: &Keypair,
    program_key: &Keypair,
    buffer_address: &Pubkey,
    elf: ProgramElfRef<'_>,
) -> Result<Slot, DeployError> {
    let payer_address = &payer.pubkey();
    let program_address = program_key.pubkey();

    let program_lamports = {
        let space = UpgradeableLoaderState::size_of_program();
        rent.minimum_balance(space)
    };

    let instructions = loader_v3_instruction::deploy_with_max_program_len(
        payer_address,
        &program_address,
        buffer_address,
        payer_address,
        program_lamports,
        elf.len(),
    )
    .map_err(DeployError::DeployWithMaxPrograLenInstructions)?;

    let transaction = Transaction::new_signed_with_payer(
        &instructions,
        Some(payer_address),
        &[payer, program_key],
        blockhash_cache.get(),
    );

    let signature = client
        .send_and_confirm_transaction(&transaction)
        .await
        .map_err(DeployError::DeployWithMaxPrograLen)?;

    debug!(
        "Buffer {buffer_address}: Called DeployWithMaxPrograLen for program at {program_address}: \
         {signature}",
    );

    loop {
        let res = client
            .get_signature_statuses(&[signature])
            .await
            .map_err(DeployError::DeployWithMaxPrograLenSignature)?;

        // Option<&Option<TransactionStatus> => Option<&TransactionStatus>
        let Some(status) = res.value.first().and_then(Option::as_ref) else {
            sleep(Duration::from_millis(100)).await;
            continue;
        };

        return Ok(status.slot);
    }
}

async fn delete_buffer(
    blockhash_cache: &BlockhashCache,
    client: &RpcClient,
    payer: &Keypair,
    buffer_address: &Pubkey,
) -> Result<(), DeployError> {
    let payer_address = &payer.pubkey();

    let close_buffer_inst =
        loader_v3_instruction::close(buffer_address, payer_address, payer_address);

    let transaction = Transaction::new_signed_with_payer(
        &[close_buffer_inst],
        Some(payer_address),
        &[payer],
        blockhash_cache.get(),
    );

    let signature = client
        .send_and_confirm_transaction(&transaction)
        .await
        .map_err(|error| DeployError::DeleteBuffer {
            address: *buffer_address,
            error,
        })?;

    debug!("Buffer {buffer_address}: Deleted: {signature}");

    Ok(())
}

pub async fn delete_program(
    blockhash_cache: &BlockhashCache,
    client: &RpcClient,
    payer: &Keypair,
    program_address: &Pubkey,
) -> Result<(), DeployError> {
    let payer_address = &payer.pubkey();

    let program_data_address =
        solana_loader_v3_interface::get_program_data_address(program_address);

    let close_program_data_inst = loader_v3_instruction::close_any(
        &program_data_address,
        payer_address,
        Some(payer_address),
        Some(program_address),
    );

    // TODO In order for the account to be fully closed, it needs to have no data.  But here seems
    // to be no built in program that can change the account data?
    //
    // I wonder why wouldn't the loader program have an instruction to close the program account?
    // It allows all 3 other types of account to be closed.
    //
    // let program_lamports =
    //     client
    //         .get_balance(program_address)
    //         .await
    //         .map_err(|error| DeployError::GetBalance {
    //             address: *program_address,
    //             error,
    //         })?;
    //
    // let close_program_inst =
    //     system_instruction::transfer(program_address, payer_address, program_lamports);
    //
    // let message = Message::new(
    //     &[close_program_data_inst, close_program_inst],
    //     Some(payer_address),
    // );

    let transaction = Transaction::new_signed_with_payer(
        &[close_program_data_inst],
        Some(payer_address),
        &[payer],
        blockhash_cache.get(),
    );

    let signature = client
        .send_and_confirm_transaction(&transaction)
        .await
        .map_err(|error| DeployError::DeleteProgram {
            address: *program_address,
            error,
        })?;

    debug!("Program {program_address}: Deleted: {signature}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use {
        super::{create_buffer, delete_buffer, delete_program, deploy, fill_buffer, finalize},
        crate::{
            blockhash_cache::BlockhashCache,
            programs::{program_elf, KnownPrograms, ProgramElfRef},
        },
        bincode,
        solana_account::Account,
        solana_clock::{Epoch, Slot},
        solana_keypair::Keypair,
        solana_loader_v3_interface::state::UpgradeableLoaderState,
        solana_pubkey::Pubkey,
        solana_rent::Rent,
        solana_rpc_client::nonblocking::rpc_client::RpcClient,
        solana_rpc_client_api::{client_error as rpc_client_error, request::RpcError},
        solana_sdk_ids::bpf_loader_upgradeable,
        solana_signer::Signer,
        solana_test_validator::TestValidatorGenesis,
        std::cmp::min,
        tokio,
    };

    // TODO Not `#[track_caller]` is not supported for async functions on stable yet.
    // https://github.com/rust-lang/rust/issues/110011
    // #[track_caller]
    async fn verify_program_buffer_state(
        rpc_client: &RpcClient,
        address: Pubkey,
        authority_address: Pubkey,
        elf: ProgramElfRef<'_>,
    ) {
        let expected_size = UpgradeableLoaderState::size_of_buffer(elf.len());
        let expected_lamports = rpc_client
            .get_minimum_balance_for_rent_exemption(expected_size)
            .await
            .unwrap();
        let expected_data = {
            let mut buf = Vec::with_capacity(expected_size);
            bincode::serialize_into(
                &mut buf,
                &UpgradeableLoaderState::Buffer {
                    authority_address: Some(authority_address),
                },
            )
            .unwrap();
            buf.extend_from_slice(elf);
            buf
        };

        let account = rpc_client.get_account(&address).await.unwrap();
        assert_eq!(account.lamports, expected_lamports);
        assert_eq!(account.data, expected_data);
        assert_eq!(account.owner, bpf_loader_upgradeable::ID);
        assert!(!account.executable);
        assert_eq!(account.rent_epoch, Epoch::MAX);
    }

    // TODO Not `#[track_caller]` is not supported for async functions on stable yet.
    // https://github.com/rust-lang/rust/issues/110011
    // #[track_caller]
    async fn verify_program_main_account_state(rpc_client: &RpcClient, address: Pubkey) {
        let program_data_address = solana_loader_v3_interface::get_program_data_address(&address);

        let expected_size = UpgradeableLoaderState::size_of_program();
        let expected_lamports = rpc_client
            .get_minimum_balance_for_rent_exemption(expected_size)
            .await
            .unwrap();
        let expected_data = bincode::serialize(&UpgradeableLoaderState::Program {
            programdata_address: program_data_address,
        })
        .unwrap();

        let account = rpc_client.get_account(&address).await.unwrap();
        assert_eq!(account.lamports, expected_lamports);
        assert_eq!(account.data, expected_data);
        assert_eq!(account.owner, bpf_loader_upgradeable::ID);
        assert!(account.executable);
        assert_eq!(account.rent_epoch, Epoch::MAX);
    }

    // TODO `#[track_caller]` is not supported for async functions on stable yet.
    // https://github.com/rust-lang/rust/issues/110011
    // #[track_caller]
    async fn verify_program_data_account_state(
        rpc_client: &RpcClient,
        address: Pubkey,
        modified_slot: Slot,
        upgrade_authority_address: Pubkey,
        elf: ProgramElfRef<'_>,
    ) {
        let program_data_address = solana_loader_v3_interface::get_program_data_address(&address);

        let expected_size = UpgradeableLoaderState::size_of_programdata(elf.len());
        let expected_lamports = rpc_client
            .get_minimum_balance_for_rent_exemption(expected_size)
            .await
            .unwrap();

        let expected_data = {
            let mut buf = Vec::with_capacity(expected_size);
            bincode::serialize_into(
                &mut buf,
                &UpgradeableLoaderState::ProgramData {
                    slot: modified_slot,
                    upgrade_authority_address: Some(upgrade_authority_address),
                },
            )
            .unwrap();
            buf.extend_from_slice(elf);
            buf
        };

        let account = rpc_client.get_account(&program_data_address).await.unwrap();
        assert_eq!(account.lamports, expected_lamports);
        assert_eq!(account.data, expected_data);
        assert_eq!(account.owner, bpf_loader_upgradeable::ID);
        assert!(!account.executable);
        assert_eq!(account.rent_epoch, Epoch::MAX);
    }

    // TODO Not `#[track_caller]` is not supported for async functions on stable yet.
    // https://github.com/rust-lang/rust/issues/110011
    // #[track_caller]
    async fn verify_account_absent(rpc_client: &RpcClient, address: Pubkey) {
        match rpc_client.get_account(&address).await {
            Ok(account) => {
                let Account {
                    lamports,
                    data,
                    owner,
                    executable,
                    rent_epoch: _,
                } = account;

                let data_len = data.len();
                let data_start = &data[0..min(data_len, 100)];
                panic!(
                    "Expected account at {address} to be absent.\nGot:\nlamports: \
                     {lamports}\ndata[0..100] of {data_len}: {data_start:?}\nowner: \
                     {owner}\nexecutable: {executable}",
                );
            }
            Err(error) => {
                let rpc_client_error::ErrorKind::RpcError(error) = error.kind() else {
                    panic!("get_account({address}) failed: {error}");
                };

                let RpcError::ForUser(error_text) = error else {
                    panic!("get_account({address}) failed: RpcError({error:?})");
                };

                assert_eq!(
                    error_text.as_str(),
                    &format!("AccountNotFound: pubkey={address}"),
                    "get_account({address}) failed: RpcError(ForUser(\"{error_text}\"))",
                );
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn simple_create_buffer() {
        let (test_validator, payer) = TestValidatorGenesis::default().start_async().await;
        let payer_address = payer.pubkey();
        let blockhash_cache = BlockhashCache::uninitialized();
        let rent = Rent::default();

        let rpc_client = test_validator.get_async_rpc_client();

        let elf = program_elf(KnownPrograms::Noop);

        blockhash_cache.refresh(&rpc_client).await.unwrap();
        let buffer_key = create_buffer(&blockhash_cache, &rent, &rpc_client, &payer, &elf)
            .await
            .unwrap();

        let expected_elf = vec![0; elf.len()];
        verify_program_buffer_state(
            &rpc_client,
            buffer_key.pubkey(),
            payer_address,
            &expected_elf,
        )
        .await;
    }

    // Besides checking basic functionality of the `fill_buffer()` call, this test can also be used
    // to make sure that the `UPLOAD_CHUNK_SIZE` constant inside `fill_buffer()` is set to the
    // maximum possible value.
    #[tokio::test(flavor = "multi_thread")]
    async fn simple_fill_buffer() {
        let (test_validator, payer) = TestValidatorGenesis::default().start_async().await;
        let payer_address = payer.pubkey();
        let blockhash_cache = BlockhashCache::uninitialized();
        let rent = Rent::default();

        let rpc_client = test_validator.get_async_rpc_client();

        let elf = program_elf(KnownPrograms::Noop);

        blockhash_cache.refresh(&rpc_client).await.unwrap();
        let buffer_key = create_buffer(&blockhash_cache, &rent, &rpc_client, &payer, &elf)
            .await
            .unwrap();
        fill_buffer(
            &blockhash_cache,
            &rpc_client,
            &payer,
            &buffer_key.pubkey(),
            &elf,
        )
        .await
        .unwrap();

        verify_program_buffer_state(&rpc_client, buffer_key.pubkey(), payer_address, &elf).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn simple_delete_buffer() {
        let (test_validator, payer) = TestValidatorGenesis::default().start_async().await;
        let payer_address = payer.pubkey();
        let blockhash_cache = BlockhashCache::uninitialized();
        let rent = Rent::default();

        let rpc_client = test_validator.get_async_rpc_client();

        let elf = program_elf(KnownPrograms::Noop);

        blockhash_cache.refresh(&rpc_client).await.unwrap();
        let buffer_key = create_buffer(&blockhash_cache, &rent, &rpc_client, &payer, &elf)
            .await
            .unwrap();
        let buffer_address = buffer_key.pubkey();

        let expected_elf = vec![0; elf.len()];
        verify_program_buffer_state(
            &rpc_client,
            buffer_key.pubkey(),
            payer_address,
            &expected_elf,
        )
        .await;

        delete_buffer(&blockhash_cache, &rpc_client, &payer, &buffer_address)
            .await
            .unwrap();

        verify_account_absent(&rpc_client, buffer_address).await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn simple_finalize() {
        let (test_validator, payer) = TestValidatorGenesis::default().start_async().await;
        let payer_address = payer.pubkey();
        let blockhash_cache = BlockhashCache::uninitialized();
        let rent = Rent::default();

        let rpc_client = test_validator.get_async_rpc_client();

        let elf = program_elf(KnownPrograms::Noop);

        let program_key = Keypair::new();
        let program_address = program_key.pubkey();

        blockhash_cache.refresh(&rpc_client).await.unwrap();
        let buffer_key = create_buffer(&blockhash_cache, &rent, &rpc_client, &payer, &elf)
            .await
            .unwrap();
        let buffer_address = buffer_key.pubkey();
        fill_buffer(&blockhash_cache, &rpc_client, &payer, &buffer_address, &elf)
            .await
            .unwrap();

        let program_deploy_slot = finalize(
            &blockhash_cache,
            &rent,
            &rpc_client,
            &payer,
            &program_key,
            &buffer_address,
            &elf,
        )
        .await
        .unwrap();

        verify_account_absent(&rpc_client, buffer_address).await;
        verify_program_main_account_state(&rpc_client, program_address).await;
        verify_program_data_account_state(
            &rpc_client,
            program_address,
            program_deploy_slot,
            payer_address,
            &elf,
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn simple_deploy() {
        let (test_validator, payer) = TestValidatorGenesis::default().start_async().await;
        let payer_address = payer.pubkey();
        let blockhash_cache = BlockhashCache::uninitialized();
        let rent = Rent::default();

        let rpc_client = test_validator.get_async_rpc_client();

        let elf = program_elf(KnownPrograms::Noop);

        let program_key = Keypair::new();
        let program_address = program_key.pubkey();

        blockhash_cache.refresh(&rpc_client).await.unwrap();
        let program_deploy_slot = deploy(
            &blockhash_cache,
            &rent,
            &rpc_client,
            payer,
            program_key,
            &elf,
        )
        .await
        .unwrap();

        verify_program_main_account_state(&rpc_client, program_address).await;
        verify_program_data_account_state(
            &rpc_client,
            program_address,
            program_deploy_slot,
            payer_address,
            &elf,
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn simple_delete_program() {
        let (test_validator, payer) = TestValidatorGenesis::default().start_async().await;
        let payer_address = payer.pubkey();
        let blockhash_cache = BlockhashCache::uninitialized();
        let rent = Rent::default();

        let rpc_client = test_validator.get_async_rpc_client();

        let elf = program_elf(KnownPrograms::Noop);

        let program_key = Keypair::new();
        let program_address = program_key.pubkey();

        blockhash_cache.refresh(&rpc_client).await.unwrap();
        let program_deploy_slot = deploy(
            &blockhash_cache,
            &rent,
            &rpc_client,
            payer.insecure_clone(),
            program_key,
            &elf,
        )
        .await
        .unwrap();
        let slot_after_program_deployment = rpc_client.get_slot().await.unwrap();

        verify_program_main_account_state(&rpc_client, program_address).await;
        verify_program_data_account_state(
            &rpc_client,
            program_address,
            program_deploy_slot,
            payer_address,
            &elf,
        )
        .await;

        // Program can not be removed in the same slot in which it was deployed :(
        loop {
            let slot = rpc_client.get_slot().await.unwrap();
            if slot != slot_after_program_deployment {
                break;
            }
        }

        delete_program(&blockhash_cache, &rpc_client, &payer, &program_address)
            .await
            .unwrap();

        // TODO `delete_program()` does not close the program account at the moment.  See
        // `delete_program()` body for additional details.
        // verify_account_absent(&rpc_client, program_address).await;

        let program_data_address =
            solana_loader_v3_interface::get_program_data_address(&program_address);
        verify_account_absent(&rpc_client, program_data_address).await;
    }
}
