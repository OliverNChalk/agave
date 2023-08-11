//! Groups of attacks which are using artificially generated blocks
//! to investigate the effect on the replay stage performance.

use {enum_iterator::Sequence, strum_macros::Display};
pub const ID: &str = "replay_stage_attack";
adversarial_feature_impl!(ReplayStageAttack);

#[derive(Clone, Debug, Display, Eq, PartialEq, Sequence, serde::Serialize, serde::Deserialize)]
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

#[derive(Clone, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdversarialConfig {
    pub selected_attack: Option<Attack>,
}
