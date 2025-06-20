use {
    solana_compute_budget_interface::ComputeBudgetInstruction, solana_hash::Hash,
    solana_instruction::Instruction, solana_keypair::Keypair, solana_signer::Signer,
    solana_transaction::Transaction,
};

pub(crate) fn create_serialized_signed_transaction(
    payer: &Keypair,
    recent_blockhash: Hash,
    mut instructions: Vec<Instruction>,
    additional_signers: Vec<&Keypair>,
    transaction_cu_budget: u32,
) -> Vec<u8> {
    let set_cu_instruction =
        ComputeBudgetInstruction::set_compute_unit_limit(transaction_cu_budget);

    // set cu instruction must be the first instruction in the transaction.
    instructions.insert(0, set_cu_instruction);

    let mut signers = vec![payer];
    signers.extend(additional_signers);

    let tx = Transaction::new_signed_with_payer(
        &instructions,
        Some(&payer.pubkey()),
        &signers,
        recent_blockhash,
    );

    bincode::serialize(&tx).expect("serialize Transaction in send_batch")
}
