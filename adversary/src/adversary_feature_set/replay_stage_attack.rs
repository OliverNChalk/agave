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
    serde::{Deserialize, Serialize},
    strum::VariantNames,
    strum_macros::{AsRefStr, Display, EnumString, EnumVariantNames},
};

mod attack_program_config;
mod config_verifiers;
mod large_nop_attack_config;
mod non_existent_accounts_attack_config;

pub use self::{
    attack_program_config::AttackProgramConfig,
    config_verifiers::{AttackConfigVerifier, VerifierRegistrationError},
    large_nop_attack_config::LargeNopAttackConfig,
    non_existent_accounts_attack_config::NonExistentAccountsAttackConfig,
};

pub const ID: &str = "replay_stage_attack";
adversarial_feature_impl!(ReplayStageAttack);

#[derive(
    Clone,
    Debug,
    Display,
    Eq,
    AsRefStr,
    EnumString,
    EnumVariantNames,
    PartialEq,
    Serialize,
    Deserialize,
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
    ReadNonExistentAccounts(NonExistentAccountsAttackConfig),
    CpiProgram(AttackProgramConfig),
}

#[derive(Default)]
pub struct AttackSubtypeStatsId(i64);

impl From<AttackSubtypeStatsId> for i64 {
    fn from(stats_id: AttackSubtypeStatsId) -> i64 {
        stats_id.0
    }
}

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
        config_verifiers::register(Self::VARIANTS, name, location, verifier)
    }

    /// When all attack verifiers are registered, call this function to make sure that no
    /// verifiers are missing.
    ///
    /// Helps detect missing verifiers early, rather then when an attempt is made to verify a
    /// specific attack configuration and the verifier is not found.
    pub fn end_verifier_registration() -> Result<(), VerifierRegistrationError> {
        config_verifiers::end_registration(Self::VARIANTS)
    }

    pub fn verify(&self, accounts: &AccountsFile) -> Result<(), String> {
        config_verifiers::verify(self.as_ref(), self, accounts)
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
            Attack::ReadNonExistentAccounts(_) => 14,
            Attack::CpiProgram(_) => 15,
        };

        AttackSubtypeStatsId(id)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdversarialConfig {
    pub selected_attack: Option<Attack>,
}
