//! Rust-based SBF that used for testing with transaction-bench
use {
    borsh::{BorshDeserialize, BorshSerialize},
    num_derive::{FromPrimitive, ToPrimitive},
    solana_account_info::AccountInfo,
    solana_msg::msg,
    solana_program_entrypoint::{entrypoint, ProgramResult},
    solana_program_error::ProgramError,
    solana_pubkey::Pubkey,
    std::ptr::read_volatile,
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
pub enum ClientTestProgramInstruction {
    ReadAccounts {
        // this is random number to avoid having duplicate transactions errors
        random: u64,
    },
}

entrypoint!(process_instruction);

pub fn process_instruction(
    _program_id: &Pubkey,
    accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    let instruction =
        ClientTestProgramInstruction::try_from_slice(instruction_data).map_err(|err| {
            msg!("Instruction deserialize failed: {}", err);
            Error::InvalidInstructionData
        })?;

    match instruction {
        ClientTestProgramInstruction::ReadAccounts { random: _ } => read_accounts(accounts),
    }
}

fn read_accounts(accounts: &[AccountInfo]) -> ProgramResult {
    for account in accounts {
        let data = &account.data;
        let first_byte = data.borrow().as_ptr();
        unsafe {
            // Make sure the compiler will emit the memory read, forcing the VM to access the
            // account data.
            let _ = read_volatile(first_byte);
        }
    }
    Ok(())
}
