//! Defines the transaction types and generates a sequence of transactions.
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

const TRANSACTION_SAMPLE_SIZE: usize = 100;
const SEED: u64 = 42;

#[derive(Clone, Copy, Debug)]
pub(crate) enum TransactionType {
    Read,
    Transfer,
    Mint,
}

#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn generate_tx_type_sequence(
    read_pct: usize,
    transfer_pct: usize,
    mint_pct: usize,
) -> Vec<TransactionType> {
    assert_eq!(
        read_pct + transfer_pct + mint_pct,
        100,
        "Mix percentages must sum to 100"
    );

    let mut sequence = Vec::with_capacity(TRANSACTION_SAMPLE_SIZE);
    for _ in 0..(TRANSACTION_SAMPLE_SIZE * read_pct / 100) {
        sequence.push(TransactionType::Read);
    }
    for _ in 0..(TRANSACTION_SAMPLE_SIZE * transfer_pct / 100) {
        sequence.push(TransactionType::Transfer);
    }
    for _ in 0..(TRANSACTION_SAMPLE_SIZE * mint_pct / 100) {
        sequence.push(TransactionType::Mint);
    }

    let mut rng = StdRng::seed_from_u64(SEED);
    sequence.shuffle(&mut rng);

    sequence
}
