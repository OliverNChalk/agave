//! Rust-based SBF that used for testing replay stage

#![allow(clippy::arithmetic_side_effects)]

use {
    borsh::{BorshDeserialize, BorshSerialize},
    num_derive::{FromPrimitive, ToPrimitive},
    solana_account_info::AccountInfo,
    solana_cpi::invoke,
    solana_instruction::{AccountMeta, Instruction},
    solana_msg::msg,
    solana_program_entrypoint::{entrypoint, ProgramResult},
    solana_program_error::ProgramError,
    solana_pubkey::Pubkey,
    std::ptr::read_volatile,
    thiserror::Error,
};
// Size of the random data in the Nop instruction in order to generate the
// largest valid transaction (currently 1232B).
pub const LARGE_NOP_DATA_SIZE: usize = 1017;

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
        // value to write to each of the accounts
        value: u8,
        // this is random number to avoid having duplicate transactions errors
        random: u64,
    },
    ReadAccounts {
        // this is random number to avoid having duplicate transactions errors
        random: u64,
    },
    Recurse {
        // recursion depth
        depth: u8,
        // this is random number to avoid having duplicate transactions errors
        random: u64,
    },
    Nop {
        random_data: Vec<u8>,
    },
    Cpi {
        // recursion depth
        depth: u8,
        // this is random number to avoid having duplicate transactions errors
        random: u64,
    },
}

entrypoint!(process_instruction);

pub fn process_instruction(
    program_id: &Pubkey,
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
        BlockGeneratorStressTestInstruction::ReadAccounts { random: _ } => read_accounts(accounts),
        BlockGeneratorStressTestInstruction::Recurse { depth, random } => {
            recurse(program_id, depth, random, accounts)
        }
        BlockGeneratorStressTestInstruction::Nop { random_data: _ } => Ok(()),
        BlockGeneratorStressTestInstruction::Cpi { depth, random } => {
            cpi(program_id, depth, random, accounts)
        }
    }
}

fn write_accounts(accounts: &[AccountInfo], value: u8) -> ProgramResult {
    for account in accounts {
        if account.is_writable {
            let mut data = account.data.borrow_mut();
            if data.len() > 0 {
                data[0] = value;
            }
        }
    }
    Ok(())
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

fn recurse(program_id: &Pubkey, depth: u8, random: u64, accounts: &[AccountInfo]) -> ProgramResult {
    if depth == 0 {
        return Ok(());
    }

    let accounts_meta: Vec<AccountMeta> = accounts
        .iter()
        .map(|a| AccountMeta {
            pubkey: *a.key,
            is_writable: a.is_writable,
            is_signer: a.is_signer,
        })
        .collect();

    let data = borsh::to_vec(&BlockGeneratorStressTestInstruction::Recurse {
        depth: depth.saturating_sub(1),
        random,
    })
    .map_err(|_| ProgramError::BorshIoError)?;
    let instruction = Instruction {
        program_id: *program_id,
        accounts: accounts_meta,
        data,
    };
    invoke(&instruction, accounts)
}

fn cpi(program_id: &Pubkey, depth: u8, random: u64, accounts: &[AccountInfo]) -> ProgramResult {
    if depth == 0 {
        return Ok(());
    }

    let accounts_meta: Vec<AccountMeta> = accounts
        .iter()
        .map(|a| AccountMeta {
            pubkey: *a.key,
            is_writable: a.is_writable,
            is_signer: a.is_signer,
        })
        .collect();

    // Make excessive CPI calls to stress the system
    let data = borsh::to_vec(&BlockGeneratorStressTestInstruction::Nop {
        random_data: vec![],
    })
    .map_err(|_| ProgramError::BorshIoError)?;
    let instruction = Instruction {
        program_id: *program_id,
        accounts: accounts_meta.clone(),
        data,
    };

    // This value was chosen to not exceed the max instruction trace length
    // and encounter MaxInstructionTraceLengthExceeded errors.
    for _ in 0..14 {
        invoke(&instruction, accounts)?;
    }

    // Make recursive CPI call to stress the system
    let data = borsh::to_vec(&BlockGeneratorStressTestInstruction::Recurse {
        depth: depth.saturating_sub(1),
        random,
    })
    .map_err(|_| ProgramError::BorshIoError)?;
    let instruction = Instruction {
        program_id: *program_id,
        accounts: accounts_meta,
        data,
    };
    invoke(&instruction, accounts)
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

    // No test_sanity_read() because there are not observable side-effects of execution this function
}
