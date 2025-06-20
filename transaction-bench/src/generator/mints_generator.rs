use {
    crate::{
        cli::MintTxParams,
        generator::{
            transaction_batch_utils::spawn_blocking_transaction_batch_generation,
            transaction_builder::create_serialized_signed_transaction,
        },
    },
    solana_hash::Hash,
    solana_instruction::Instruction,
    solana_keypair::Keypair,
    solana_program_pack::Pack,
    solana_rent::Rent,
    solana_signer::Signer,
    solana_system_interface::instruction as system_instruction,
    spl_associated_token_account_interface::{
        address::get_associated_token_address, instruction::create_associated_token_account,
    },
    spl_token_interface::{instruction as spl_token_instruction, state::Mint},
    std::sync::Arc,
    tokio::task::JoinHandle,
};

// Generates a transaction batch of mint transactions.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn generate_mint_batch(
    payers: Arc<Vec<Keypair>>,
    mut payer_index: usize,
    blockhash: Hash,
    params: MintTxParams,
    send_batch_size: usize,
) -> JoinHandle<Vec<Vec<u8>>> {
    spawn_blocking_transaction_batch_generation("generate mint transaction batch", move || {
        let mut txs: Vec<Vec<u8>> = Vec::with_capacity(send_batch_size);
        for _ in 0..send_batch_size {
            let payer = &payers[payer_index];
            payer_index = (payer_index + 1) % payers.len();

            let mint_account = Keypair::new();
            let mint_authority = Keypair::new();

            let tx = create_serialized_signed_transaction(
                payer,
                blockhash,
                create_token_mint_instructions(
                    payer,
                    &mint_account,
                    &mint_authority,
                    params.decimals,
                    params.initial_supply,
                ),
                vec![&mint_account, &mint_authority],
                params.mint_tx_cu_budget,
            );

            txs.push(tx);
        }
        txs
    })
}

/// Creates a set of instructions that mint a new token.
fn create_token_mint_instructions(
    payer: &Keypair,
    mint_account: &Keypair,
    mint_authority: &Keypair,
    decimals: u8,
    initial_supply: u64,
) -> Vec<Instruction> {
    // Instruction 0: Set the compute unit limit - will be handled by create_serialized_signed_transaction().

    // Instruction 1: Create the mint account.
    let mint_space = Mint::LEN;
    let lamports = Rent::default().minimum_balance(mint_space);
    let create_mint_account_ix = system_instruction::create_account(
        &payer.pubkey(),
        &mint_account.pubkey(),
        lamports,
        mint_space as u64,
        &spl_token_interface::id(),
    );

    // Instruction 2: Initialize the mint account.
    let initialize_mint_ix = spl_token_instruction::initialize_mint(
        &spl_token_interface::id(),
        &mint_account.pubkey(),
        &mint_authority.pubkey(),
        None,
        decimals,
    )
    .expect("Failed to initialize mint");

    // Instruction 3: Create the token account.
    let create_ata_ix = create_associated_token_account(
        &payer.pubkey(),
        &payer.pubkey(),
        &mint_account.pubkey(),
        &spl_token_interface::id(),
    );

    // Instruction 4: Mint the initial supply of tokens to the token account.
    let associated_token_account =
        get_associated_token_address(&payer.pubkey(), &mint_account.pubkey());
    let mint_tokens_ix = spl_token_instruction::mint_to(
        &spl_token_interface::id(),
        &mint_account.pubkey(),
        &associated_token_account,
        &mint_authority.pubkey(),
        &[],
        initial_supply,
    )
    .expect("Failed to mint tokens");

    vec![
        create_mint_account_ix,
        initialize_mint_ix,
        create_ata_ix,
        mint_tokens_ix,
    ]
}
