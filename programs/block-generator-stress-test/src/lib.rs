//! Rust-based SBF that used for testing replay stage

#![allow(clippy::arithmetic_side_effects)]

use {
    borsh::{BorshDeserialize, BorshSerialize},
    num_derive::{FromPrimitive, ToPrimitive},
    solana_account_info::AccountInfo,
    solana_msg::msg,
    solana_program_entrypoint::{entrypoint, ProgramResult},
    solana_program_error::ProgramError,
    solana_pubkey::Pubkey,
    thiserror::Error,
};

#[derive(Error, Debug, Clone, PartialEq, Eq, FromPrimitive, ToPrimitive)]
pub enum Error {
    #[error("Failed to deserialize instruction data")]
    InvalidInstructionData,
}

impl From<Error> for ProgramError {
    fn from(e: Error) -> Self {
        ProgramError::Custom(e as u32)
    }
}

// Enum describing possible entry point instructions
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize, PartialEq)]
pub enum BlockGeneratorStressTestInstruction {
    WriteAccounts {
        value: u8,   // value to write to each of the accounts
        random: u64, // this is random number to avoid having duplicate transactions errors
    },
}

entrypoint!(process_instruction);

pub fn process_instruction(
    _program_id: &Pubkey,
    accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    let instruction = BlockGeneratorStressTestInstruction::try_from_slice(instruction_data)
        .map_err(|err| {
            msg!("Instruction deserialize failed: {}", err);
            Error::InvalidInstructionData
        })?;

    match instruction {
        BlockGeneratorStressTestInstruction::WriteAccounts { value, random: _ } => {
            write_accounts(accounts, value)
        }
    };

    Ok(())
}

fn write_accounts(accounts: &[AccountInfo], value: u8) {
    for account in accounts {
        if account.is_writable {
            let mut data = account.data.borrow_mut();
            if data.len() > 0 {
                data[0] = value;
            }
        }
    }
}

// Sanity tests
// To run use `cargo test-sbf
#[cfg(test)]
mod test {
    use {super::*, std::mem};

    #[test]
    fn test_sanity_write() {
        let program_id = Pubkey::default();
        let key = Pubkey::default();
        let mut lamports = 0;
        let mut data = vec![0; mem::size_of::<u32>()];
        let owner = Pubkey::default();
        let writable_account =
            AccountInfo::new(&key, false, true, &mut lamports, &mut data, &owner, false);
        let mut lamports = 0;
        let mut data = vec![0; mem::size_of::<u32>()];
        let readonly_account =
            AccountInfo::new(&key, false, false, &mut lamports, &mut data, &owner, false);
        assert_eq!(writable_account.data.borrow_mut()[0], 0u8);

        let accounts = vec![writable_account.clone(), writable_account, readonly_account];
        let instruction = BlockGeneratorStressTestInstruction::WriteAccounts {
            value: 128u8,
            random: 7,
        };
        process_instruction(
            &program_id,
            &accounts,
            &borsh::to_vec(&instruction).unwrap(),
        )
        .unwrap();
        assert_eq!(accounts[0].data.borrow()[0], 128u8);
        assert_eq!(accounts[1].data.borrow()[0], 128u8);
        assert_eq!(accounts[2].data.borrow()[0], 0u8);
    }
}
