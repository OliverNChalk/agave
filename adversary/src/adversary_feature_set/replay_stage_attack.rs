//! Groups of attacks which are using artificially generated blocks to investigate the effect on the
//! replay stage performance.
//!
//! Individual attacks may have configuration and constraints on when they can be executed.  Those
//! are expressed as configuration verifiers, that need to be registered via an
//! [`Attacks::register_verifier()`] call.  This is effectively a late binding.
//!
//! It would be nice to be able to call individual attack config verifiers directly.  But they are
//! defined in the `solana_core` crate, and `adversary` crate can not directly depend on it.  So we
//! have to use late binding.
//!
//! We may want to replace [`Attack`] with an open enum, that is populated in a way similar to the
//! current verifiers setup.  Making both configuration and execution of the attack fully defined in
//! a single location, removing all attack specific code from this module.

use {
    crate::accounts_file::AccountsFile,
    block_generator_stress_test::LARGE_NOP_DATA_SIZE,
    itertools::Itertools,
    std::{
        collections::{hash_map, HashMap},
        sync::{LazyLock, Mutex},
    },
    strum::VariantNames,
    strum_macros::{AsRefStr, Display, EnumString, EnumVariantNames},
    thiserror::Error,
};

pub const ID: &str = "replay_stage_attack";
adversarial_feature_impl!(ReplayStageAttack);

#[derive(Copy, Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AttackProgramConfig {
    /// Max value is 64. In some parts of the code it is called "entry size".
    pub transaction_batch_size: usize,
    pub num_accounts_per_tx: usize,
    pub transaction_cu_budget: u32,
    /// Attacks involving expensive computations might be configured with
    /// option to bypass execution. For that, they must be configured to fail.
    /// This might be achieved by requesting less `transaction_cu_budget` than required
    pub use_failed_transaction_hotpath: bool,
}

// Default values are such that generated block can be replayed in ~400ms.
// Generating heavier blocks is possible but requires skipping loading accounts and execution
// transactions in the block.
impl Default for AttackProgramConfig {
    fn default() -> Self {
        Self {
            transaction_batch_size: 1,
            num_accounts_per_tx: 1,
            // high enough value so that transaction is valid
            transaction_cu_budget: 1_000,
            use_failed_transaction_hotpath: false,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
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

#[derive(
    Clone,
    Debug,
    Display,
    Eq,
    AsRefStr,
    EnumString,
    EnumVariantNames,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(rename_all = "camelCase")]
#[strum(serialize_all = "camelCase")]
pub enum Attack {
    TransferRandom,
    CreateNonceAccounts,
    AllocateRandomLarge,
    AllocateRandomSmall,
    ChainTransactions,
    WriteProgram(AttackProgramConfig),
    ReadMaxAccounts,
    WriteMaxAccounts,
    ReadProgram(AttackProgramConfig),
    RecursiveProgram(AttackProgramConfig),
    ColdProgramCache(AttackProgramConfig),
    LargeNop(LargeNopAttackConfig),
    TransferRandomWithMemo,
    ReadNonExistentAccounts,
}

#[derive(Default)]
pub struct AttackSubtypeStatsId(i64);

impl From<AttackSubtypeStatsId> for i64 {
    fn from(stats_id: AttackSubtypeStatsId) -> i64 {
        stats_id.0
    }
}

pub type AttackConfigVerifier =
    Box<dyn Fn(&AccountsFile, &Attack) -> Result<(), String> + Send + 'static>;

#[derive(Error, Debug)]
pub enum VerifierRegistrationError {
    #[error("No Attack member exists named '{0}'.")]
    UnexpectedName(&'static str),

    #[error("Entry for '{0}' is already registered at: {1}")]
    AlreadyRegistered(&'static str, &'static str),

    #[error("Verifier for '{0}' was not registered along with the rest of the verifiers.")]
    Missing(&'static str),
}

struct VerifierInfo {
    // Source location of the verifier registration or the verifier function itself.
    // To allow duplicate registration of the save verifier.
    location: &'static str,

    verifier: AttackConfigVerifier,
}

static CONFIG_VERIFIERS: LazyLock<Mutex<HashMap<&'static str, VerifierInfo>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

impl Attack {
    pub const fn cli_names() -> &'static [&'static str] {
        Self::VARIANTS
    }

    /// Register a config verifier for the specified attack `name`, as listed in
    /// [`Attack::VARIANTS`].
    ///
    /// Verifier should be registered for a given attack only once.  But some tests create the TPU
    /// multiple times.  In order to distinguish a case when the same verifier is registered twice
    /// from the case when two different verifiers are registered for the same attack, `location` is
    /// used.  Duplicate registration for the same attack with an identical `location` is ignored.
    ///
    /// `location` should be a file and line of the verifier registration or the verifier function.
    pub fn register_config_verifier(
        name: &'static str,
        location: &'static str,
        verifier: AttackConfigVerifier,
    ) -> Result<(), VerifierRegistrationError> {
        if !Self::VARIANTS.contains(&name) {
            return Err(VerifierRegistrationError::UnexpectedName(name));
        }

        let mut verifiers = CONFIG_VERIFIERS.lock().unwrap();
        let entry = match verifiers.entry(name) {
            hash_map::Entry::Occupied(entry) => {
                let existing_location = entry.get().location;
                if location == existing_location {
                    // Duplicate registration of the same verifier.
                    return Ok(());
                } else {
                    return Err(VerifierRegistrationError::AlreadyRegistered(
                        name,
                        existing_location,
                    ));
                }
            }
            hash_map::Entry::Vacant(entry) => entry,
        };

        entry.insert(VerifierInfo { location, verifier });
        Ok(())
    }

    /// When all attack verifiers are registered, call this function to make sure that no
    /// verifiers are missing.
    ///
    /// Helps detect missing verifiers early, rather then when an attempt is made to verify a
    /// specific attack configuration and the verifier is not found.
    pub fn end_verifier_registration() -> Result<(), VerifierRegistrationError> {
        let verifiers = CONFIG_VERIFIERS.lock().unwrap();
        for name in Self::VARIANTS {
            if !verifiers.contains_key(name) {
                return Err(VerifierRegistrationError::Missing(name));
            }
        }

        Ok(())
    }

    pub fn verify(&self, accounts: &AccountsFile) -> Result<(), String> {
        let verifiers = CONFIG_VERIFIERS.lock().unwrap();
        let verifier = {
            let name = self.as_ref();
            &verifiers
                .get(name)
                .unwrap_or_else(|| {
                    let has_verifiers_for = if verifiers.is_empty() {
                        "<none>".to_owned()
                    } else {
                        // When `Iterator::intersperse` is actually added to the stable part of the
                        // standard library, then we can decide if we want to switch, or if we want
                        // to keep using `Itertools::intersperse()`.  There seems to be no value in
                        // this warning before then.
                        #[allow(unstable_name_collisions)]
                        verifiers
                            .keys()
                            .copied()
                            .intersperse(", ")
                            .collect::<String>()
                    };

                    panic!(
                        "All attack verifiers should be set before the first RPC call.\nMake sure \
                         that `core::banking_stage::adversary::register_attack_config_verifiers()` \
                         has been invoked.\nMissing config verifier for: {name}\nVerifiers have \
                         been registered for:\n{has_verifiers_for}"
                    )
                })
                .verifier
        };

        verifier(accounts, self)
    }

    pub fn stats_id(&self) -> AttackSubtypeStatsId {
        let id = match self {
            Attack::TransferRandom => 1,
            Attack::CreateNonceAccounts => 2,
            Attack::AllocateRandomLarge => 3,
            Attack::AllocateRandomSmall => 4,
            Attack::ChainTransactions => 5,
            Attack::WriteProgram(_) => 6,
            Attack::ReadMaxAccounts => 7,
            Attack::WriteMaxAccounts => 8,
            Attack::ReadProgram(_) => 9,
            Attack::RecursiveProgram(_) => 10,
            Attack::ColdProgramCache(_) => 11,
            Attack::LargeNop(_) => 12,
            Attack::TransferRandomWithMemo => 13,
            Attack::ReadNonExistentAccounts => 14,
        };

        AttackSubtypeStatsId(id)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdversarialConfig {
    pub selected_attack: Option<Attack>,
}
