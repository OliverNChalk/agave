//! Utilities for creating accounts and programs for the local cluster.
//!
//! Note this is _not_ meant to be a general utility for creating accounts and programs.
//! This is specifically tailored to the needs of creating accounts and programs for the
//! local cluster. The data structures defined herein are laid out specifically to work with
//! the [`AccountsFile`] and the `starting_accounts` parameter of the
//! [`crate::local_cluster::ClusterConfig`] constructor.
use {
    indoc::formatdoc,
    solana_account::{Account, AccountSharedData},
    solana_adversary::accounts_file::AccountsFile,
    solana_clock::Epoch,
    solana_keypair::Keypair,
    solana_pubkey::Pubkey,
    solana_rent::Rent,
    solana_sdk_ids::bpf_loader,
    solana_signer::Signer,
    std::{iter, sync::LazyLock},
    tempfile::tempdir,
};

pub const STATIC_STRESS_TEST_PROGRAM_ID: Pubkey = Pubkey::new_from_array([7u8; 32]);

/// Compiles block-generator-stress-test-program in the temporary directory and
/// returns content of the generated so file.
/// Specifically, it runs the following:
/// cargo run --bin cargo-build-sbf --
///  --sbf-sdk "platform-tools-sdk/sbf"
///  --sbf-out-dir local-cluster/tests/program/
///  --manifest-path programs/block-generator-stress-test/Cargo.toml
fn compile_stress_test_program() -> Vec<u8> {
    // create a directory inside of std::env::temp_dir(), removed when goes out of scope
    let target_directory = tempdir().expect("temporary folder should be created");

    let manifest_directory = std::path::PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
    let source_directory = manifest_directory.join("..");

    let mut binding = std::process::Command::new(std::env!("CARGO"));
    let command = binding.current_dir(&source_directory).arg("run");

    if !cfg!(debug_assertions) {
        command.arg("--release");
    };

    command.args([
        "--bin",
        "cargo-build-sbf",
        "--",
        "--sbf-sdk",
        "platform-tools-sdk/sbf",
        "--sbf-out-dir",
        target_directory.path().to_str().unwrap(),
        "--manifest-path",
        "programs/block-generator-stress-test/Cargo.toml",
    ]);

    let output = command
        .output()
        .expect("block-generator-stress-test program should be successfully compiled");

    if !output.status.success() {
        let details = formatdoc! {"
                    Command: {:?}
                    Exit status: {:?}
                    std output:
                    {}
                    std error:
                    {}
                    ",
            command,
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        };
        panic!("block-generator-stress-test program compilation failed:\n{details}");
    }

    let target_file_name = "block_generator_stress_test.so";
    let target_path_name = target_directory.path().join(target_file_name);
    std::fs::read(target_path_name)
        .expect("Failed to read the program file.\nPath: {target_file_name}")
}

static STRESS_TEST_PROGRAM_BYTES: LazyLock<Vec<u8>> = LazyLock::new(compile_stress_test_program);

pub struct CreateAccountConfig<'a> {
    pub num_accounts: usize,
    pub lamports_per_account: u64,
    pub space: usize,
    pub owner: &'a Pubkey,
}

/// A collection of programs to be deployed to the cluster.
///
/// Note that the layout of this struct is intentionally chosen to accommodate both
/// the [`AccountsFile`] constructor and the `starting_accounts` parameter of the
/// [`crate::local_cluster::ClusterConfig`] constructor.
#[derive(Default)]
struct TestPrograms {
    pub program_ids: Vec<Pubkey>,
    pub accounts: Vec<(Pubkey, AccountSharedData)>,
}

/// A collection of accounts to be deployed to the cluster.
///
/// Note that the layout of this struct is intentionally chosen to accommodate both
/// the [`AccountsFile`] constructor and the `starting_accounts` parameter of the
/// [`crate::local_cluster::ClusterConfig`] constructor.
#[derive(Default)]
struct TestAccounts {
    pub keypairs: Vec<Keypair>,
    pub accounts: Vec<(Pubkey, AccountSharedData)>,
}

fn create_stress_test_account_shared_data() -> AccountSharedData {
    AccountSharedData::from(Account {
        lamports: Rent::default()
            .minimum_balance(STRESS_TEST_PROGRAM_BYTES.len())
            .min(1),
        data: STRESS_TEST_PROGRAM_BYTES.to_vec(),
        owner: bpf_loader::id(),
        executable: true,
        rent_epoch: Epoch::MAX,
    })
}

/// Create a single static stress test program, with a hardcoded program ID,
/// [`STATIC_STRESS_TEST_PROGRAM_ID`].
fn create_static_stress_test_program() -> TestPrograms {
    TestPrograms {
        program_ids: vec![STATIC_STRESS_TEST_PROGRAM_ID],
        accounts: vec![(
            STATIC_STRESS_TEST_PROGRAM_ID,
            create_stress_test_account_shared_data(),
        )],
    }
}

/// Create `num_programs` stress test programs with unique program IDs.
///
/// If you just want a single static stress test program, use
/// [`create_static_stress_test_program`] instead.
fn create_stress_test_programs(num_programs: usize) -> TestPrograms {
    let (program_ids, accounts) = iter::repeat_with(Keypair::new)
        .take(num_programs)
        .map(|kp| {
            let pk = kp.pubkey();
            (pk, (pk, create_stress_test_account_shared_data()))
        })
        .unzip();

    TestPrograms {
        program_ids,
        accounts,
    }
}

/// Create accounts (typically payers or max size accounts) with the given config.
fn create_simple_accounts(
    CreateAccountConfig {
        num_accounts,
        lamports_per_account,
        space,
        owner,
    }: CreateAccountConfig<'_>,
) -> TestAccounts {
    let (keypairs, accounts) = iter::repeat_with(Keypair::new)
        .take(num_accounts)
        .map(|kp| {
            let pk = kp.pubkey();
            (
                kp,
                (
                    pk,
                    AccountSharedData::new(lamports_per_account, space, owner),
                ),
            )
        })
        .unzip();

    TestAccounts { keypairs, accounts }
}

/// Utility for building up a series of test accounts and programs.
///
/// This allows incrementally building up a set of accounts and programs
/// that can ultimately be converted into an [`AccountsFile`] or collected
/// into a vector of all accounts and their [`AccountSharedData`].
///
/// All fields correspond to values expected by the [`AccountsFile`] constructor.
///
/// # Example
///
/// ```no_run
/// use {
///     solana_local_cluster::account_util::{AccountsBuilder, CreateAccountConfig},
///     solana_sdk_ids::system_program, solana_pubkey::Pubkey,
/// };
///
/// let mut accounts_builder = AccountsBuilder::default();
/// let program_id = accounts_builder.with_static_stress_test_program();
/// let max_account_owner = program_id;
/// let payers = accounts_builder.with_payers(CreateAccountConfig {
///     num_accounts: 1,
///     lamports_per_account: 1_000_000,
///     space: 0,
///     owner: &system_program::id(),
/// });
/// let max_size_accounts = accounts_builder.with_max_size_accounts(CreateAccountConfig {
///     num_accounts: 1,
///     lamports_per_account: 1_000_000,
///     space: 100_000,
///     owner: &max_account_owner,
/// });
///
/// accounts_builder.with_owner_program_id(max_account_owner);
///
/// let accounts_file = accounts_builder.as_accounts_file();
/// let all_accounts = accounts_builder.collect_accounts();
/// ```
#[derive(Default)]
pub struct AccountsBuilder {
    payers: Option<TestAccounts>,
    max_size_accounts: Option<TestAccounts>,
    programs: Option<TestPrograms>,
    pub owner_program_id: Option<Pubkey>,
}

impl AccountsBuilder {
    /// Create payers with the given config.
    pub fn with_payers(&mut self, config: CreateAccountConfig) -> &[Keypair] {
        self.payers = Some(create_simple_accounts(config));
        self.payers.as_ref().unwrap().keypairs.as_slice()
    }

    /// Create max size accounts with the given config.
    pub fn with_max_size_accounts(&mut self, config: CreateAccountConfig) -> &[Keypair] {
        self.max_size_accounts = Some(create_simple_accounts(config));
        self.max_size_accounts.as_ref().unwrap().keypairs.as_slice()
    }

    /// Create a static stress test program with a hardcoded program ID,
    /// [`STATIC_STRESS_TEST_PROGRAM_ID`].
    pub fn with_static_stress_test_program(&mut self) -> Pubkey {
        self.programs = Some(create_static_stress_test_program());
        self.programs.as_ref().unwrap().program_ids[0]
    }

    /// Create `num_programs` stress test programs.
    ///
    /// If you just want a single static stress test program, use
    /// [`AccountsBuilder::with_static_stress_test_program`] instead.
    pub fn with_stress_test_programs(&mut self, num_programs: usize) -> &[Pubkey] {
        self.programs = Some(create_stress_test_programs(num_programs));
        self.programs.as_ref().unwrap().program_ids.as_slice()
    }

    /// Create `num_programs` stress test programs.
    ///
    /// It is recommended to use the more semantically meaningful methods
    /// [`AccountsBuilder::with_static_stress_test_program`] or
    /// [`AccountsBuilder::with_stress_test_programs`] instead. This method exists solely for
    /// backwards compatibility reasons.
    ///
    /// - If `num_programs` is 0, no stress test programs are created.
    /// - If `num_programs` is 1, a static stress test program is created with program ID
    ///   [`STATIC_STRESS_TEST_PROGRAM_ID`].
    /// - Otherwise, `num_programs` stress test programs are created.
    #[deprecated(
        note = "use `with_static_stress_test_program` or `with_stress_test_programs` instead"
    )]
    pub fn with_optional_stress_test_programs(&mut self, num_programs: usize) -> Option<&[Pubkey]> {
        self.programs = match num_programs {
            0 => return None,
            1 => Some(create_static_stress_test_program()),
            n => Some(create_stress_test_programs(n)),
        };

        self.programs.as_ref().map(|p| p.program_ids.as_slice())
    }

    /// Set the owner program ID.
    pub fn with_owner_program_id(&mut self, program_id: Pubkey) {
        self.owner_program_id = Some(program_id);
    }

    /// Generate an [`AccountsFile`] from the builder.
    pub fn as_accounts_file(&self) -> AccountsFile {
        AccountsFile::new(
            self.owner_program_id,
            self.payers.as_ref().map(|a| a.keypairs.as_slice()),
            self.max_size_accounts
                .as_ref()
                .map(|a| a.keypairs.as_slice()),
            self.programs.as_ref().map(|p| p.program_ids.as_slice()),
        )
    }

    /// Collect all accounts into a single vector.
    pub fn collect_accounts(self) -> Vec<(Pubkey, AccountSharedData)> {
        [
            self.payers.map(|p| p.accounts),
            self.max_size_accounts.map(|m| m.accounts),
            self.programs.map(|p| p.accounts),
        ]
        .into_iter()
        .flatten()
        .flatten()
        .collect()
    }
}

impl From<AccountsBuilder> for AccountsFile {
    fn from(builder: AccountsBuilder) -> Self {
        builder.as_accounts_file()
    }
}

impl From<AccountsBuilder> for Vec<(Pubkey, AccountSharedData)> {
    fn from(builder: AccountsBuilder) -> Self {
        builder.collect_accounts()
    }
}
