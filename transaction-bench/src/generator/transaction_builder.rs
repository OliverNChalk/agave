use {
    client_test_program::ClientTestProgramInstruction,
    rand::{thread_rng, RngCore},
    solana_compute_budget_interface::ComputeBudgetInstruction,
    solana_hash::Hash,
    solana_instruction::{AccountMeta, Instruction},
    solana_keypair::Keypair,
    solana_program_pack::Pack,
    solana_pubkey::Pubkey,
    solana_rent::Rent,
    solana_signer::Signer,
    solana_system_interface::instruction as system_instruction,
    solana_transaction::Transaction,
    spl_associated_token_account_interface::{
        address::get_associated_token_address, instruction::create_associated_token_account,
    },
    spl_token_interface::{instruction as spl_token_instruction, state::Mint},
};

pub(crate) fn create_read_transaction(
    payer: &Keypair,
    program_id: &Pubkey,
    recent_blockhash: Hash,
    accounts_meta: Vec<AccountMeta>,
    transaction_cu_budget: u32,
) -> Transaction {
    let data = ClientTestProgramInstruction::ReadAccounts {
        random: thread_rng().next_u64(),
    };

    // Explicitly specify the CU budget to avoid dropping some transactions on the CU check side.
    // The constraint is that 48M/64 > CU limit.
    // To maximize number of txs in the block, it is beneficial to set CU limit to be close to the
    // real CU consumption.
    // Default is 200k.
    let set_cu_instruction =
        ComputeBudgetInstruction::set_compute_unit_limit(transaction_cu_budget);

    let read_instruction = Instruction::new_with_borsh(*program_id, &data, accounts_meta);

    Transaction::new_signed_with_payer(
        &[set_cu_instruction, read_instruction],
        Some(&payer.pubkey()),
        &[payer],
        recent_blockhash,
    )
}

pub(crate) fn create_transfer_transaction(
    payer: &Keypair,
    receiver: &Pubkey,
    recent_blockhash: Hash,
    transfer_tx_cu_budget: u32,
    lamports_to_transfer: u64,
) -> Transaction {
    let set_cu_instruction =
        ComputeBudgetInstruction::set_compute_unit_limit(transfer_tx_cu_budget);
    // We don't set_loaded_accounts_data_size_limit because it costs 150 CU but doesn't save any.

    let transfer_instruction =
        system_instruction::transfer(&payer.pubkey(), receiver, lamports_to_transfer);

    Transaction::new_signed_with_payer(
        &[set_cu_instruction, transfer_instruction],
        Some(&payer.pubkey()),
        &[payer],
        recent_blockhash,
    )
}

/// Creates a transaction that creates a new mint account and initializes it.
pub(crate) fn create_token_mint_transaction(
    payer: &Keypair,
    mint_account: &Keypair,
    mint_authority: &Keypair,
    recent_blockhash: Hash,
    decimals: u8,
    initial_supply: u64,
    mint_tx_cu_budget: u32,
) -> Transaction {
    // Instruction 0: Set the compute unit limit.
    let set_cu_instruction = ComputeBudgetInstruction::set_compute_unit_limit(mint_tx_cu_budget);

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

    Transaction::new_signed_with_payer(
        &[
            set_cu_instruction,
            create_mint_account_ix,
            initialize_mint_ix,
            create_ata_ix,
            mint_tokens_ix,
        ],
        Some(&payer.pubkey()),
        &[payer, mint_authority, mint_account],
        recent_blockhash,
    )
}
