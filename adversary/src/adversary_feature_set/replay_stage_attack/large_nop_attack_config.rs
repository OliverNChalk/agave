use {
    super::AttackProgramConfig,
    block_generator_stress_test::LARGE_NOP_DATA_SIZE,
    serde::{Deserialize, Serialize},
};

/// Configuration for the `LargeNop` attack.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LargeNopAttackConfig {
    pub common: AttackProgramConfig,
    pub tx_data_size: usize,
}

impl Default for LargeNopAttackConfig {
    fn default() -> Self {
        Self {
            common: AttackProgramConfig {
                // Larger batch size because we generate tx in parallel using a
                // thread pool
                transaction_batch_size: 64,
                ..Default::default()
            },
            tx_data_size: LARGE_NOP_DATA_SIZE,
        }
    }
}
