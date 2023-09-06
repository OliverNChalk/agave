//! Groups of attacks which are using artificially generated blocks
//! to investigate the effect on the replay stage performance.

use {
    strum::VariantNames,
    strum_macros::{Display, EnumString, EnumVariantNames},
};

pub const ID: &str = "replay_stage_attack";
adversarial_feature_impl!(ReplayStageAttack);

#[derive(
    Clone,
    Debug,
    Display,
    Eq,
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
    WriteProgram,
}

impl Attack {
    pub const fn cli_names() -> &'static [&'static str] {
        Self::VARIANTS
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdversarialConfig {
    pub selected_attack: Option<Attack>,
}
