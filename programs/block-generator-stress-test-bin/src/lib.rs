//! This library exists to complied and provide bytes of the
//! `solana-sbf-block-generator-stress-test` program.  The program ELF bytes are embedded into
//! library binary.
//!
//! The compilation of the `solana-sbf-block-generator-stress-test` program needs to be done
//! manually, following these steps:
//!
//! ```bash
//! ./cargo run --bin cargo-build-sbf -- \
//!     --sbf-sdk platform-tools-sdk/sbf \
//!     --sbf-out-dir programs/block-generator-stress-test-bin/bin \
//!     --manifest-path programs/block-generator-stress-test/Cargo.toml
//!
//! rm programs/block-generator-stress-test-bin/bin/block_generator_stress_test-keypair.json
//!
//! git add --updated
//! ```
//!

use std::include_bytes;

/// Bytes of the `solana-sbf-block-generator-stress-test` program.
// `block-generator-stress-test.so` is emitted in `build.rs`.
#[cfg(not(windows))]
pub const BLOCK_GENERATOR_STRESS_TEST_PROGRAM_BYTES: &[u8] =
    include_bytes!("../bin/block_generator_stress_test.so");

/// Bytes of the `solana-sbf-block-generator-stress-test` program.
// `block-generator-stress-test.so` is emitted in `build.rs`.
#[cfg(windows)]
pub const BLOCK_GENERATOR_STRESS_TEST_PROGRAM_BYTES: &[u8] =
    include_bytes!("..\\..\\block_generator_stress_test.so");
