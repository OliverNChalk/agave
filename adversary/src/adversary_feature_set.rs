//! Collection of adversarial features for "solana-invalidator".
//!
//! For details on how to add adversarial features, see the README in the root
//! of this crate.
use {
    serde::{Deserialize, Serialize},
    std::{
        collections::HashMap,
        sync::{Arc, LazyLock, RwLock},
    },
};

// Macro to implement convenience methods for getting and setting the
// configuration of adversarial features. This macro should be invoked
// within a module that contains the implementation of a feature. Namely,
// a feature-specific `ID` and `AdversarialConfig` need to be in scope.
//
// $enum_variant refers to the name of the variant for this feature in
// the `AdversaryFeatureConfig` enum.
macro_rules! adversarial_feature_impl {
    ($enum_variant:ident) => {
        fn extract_config(config: super::AdversaryFeatureConfig) -> AdversarialConfig {
            match config {
                super::AdversaryFeatureConfig::$enum_variant(config) => config,
                _ => panic!("Extracted wrong type from AdversaryFeatureConfig"),
            }
        }

        pub fn get_config() -> AdversarialConfig {
            extract_config(super::get_adversary_feature_config(ID))
        }

        pub fn set_config(config: AdversarialConfig) {
            super::set_adversary_feature_config(
                ID,
                super::AdversaryFeatureConfig::$enum_variant(config),
            );
        }
    };
}

pub mod example {
    pub const ID: &str = "example";
    adversarial_feature_impl!(Example);

    #[derive(Clone, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct AdversarialConfig {
        pub example_num: u64,
    }
}

pub mod repair_packet_flood {
    use std::net::IpAddr;
    pub const ID: &str = "repair_packet_flood";
    adversarial_feature_impl!(RepairPacketFlood);

    #[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub enum PeerIdentifier {
        Pubkey(String),
        Ip(IpAddr),
    }

    #[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub enum FloodStrategy {
        MinimalPackets,
        SignedPackets,
        PingCacheOverflow,
        Orphan,
    }

    #[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct FloodConfig {
        pub flood_strategy: FloodStrategy,
        pub packets_per_peer_per_iteration: u32,
        pub iteration_delay_us: u64,
        pub target: Option<PeerIdentifier>,
    }

    #[derive(Clone, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct AdversarialConfig {
        pub configs: Vec<FloodConfig>,
    }
}

pub mod repair_parameters {
    pub const ID: &str = "repair_parameters";
    adversarial_feature_impl!(RepairParameters);

    #[derive(Clone, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct AdversarialConfig {
        pub serve_repair_max_requests_per_iteration: Option<usize>,
        pub serve_repair_oversampled_requests_per_iteration: Option<usize>,
    }
}

pub mod shred_receiver_address {
    use std::net::SocketAddr;
    pub const ID: &str = "shred_receiver_address";
    adversarial_feature_impl!(ShredReceiverAddress);

    #[derive(Clone, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct AdversarialConfig {
        pub shred_receiver_address: Option<SocketAddr>,
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
/// Enum wrapper for all adversarial feature configuration structs
pub enum AdversaryFeatureConfig {
    #[serde(rename = "exampleAdversarialConfig")]
    Example(example::AdversarialConfig),
    #[serde(rename = "repairPacketFloodAdversarialConfig")]
    RepairPacketFlood(repair_packet_flood::AdversarialConfig),
    #[serde(rename = "repairParametersConfig")]
    RepairParameters(repair_parameters::AdversarialConfig),
    #[serde(rename = "shredReceiverAddress")]
    ShredReceiverAddress(shred_receiver_address::AdversarialConfig),
}

static FEATURE_CONFIG_MAP: LazyLock<Arc<RwLock<HashMap<String, AdversaryFeatureConfig>>>> =
    LazyLock::new(|| {
        Arc::new(RwLock::new(
            [
                (
                    example::ID.to_string(),
                    AdversaryFeatureConfig::Example(example::AdversarialConfig::default()),
                ),
                (
                    repair_packet_flood::ID.to_string(),
                    AdversaryFeatureConfig::RepairPacketFlood(
                        repair_packet_flood::AdversarialConfig::default(),
                    ),
                ),
                (
                    repair_parameters::ID.to_string(),
                    AdversaryFeatureConfig::RepairParameters(
                        repair_parameters::AdversarialConfig::default(),
                    ),
                ),
                (
                    shred_receiver_address::ID.to_string(),
                    AdversaryFeatureConfig::ShredReceiverAddress(
                        shred_receiver_address::AdversarialConfig::default(),
                    ),
                ),
            ]
            .iter()
            .cloned()
            .collect(),
        ))
    });

/// Return the current configuration of all adversarial feature.
pub fn get_adversary_feature_status() -> Vec<AdversaryFeatureConfig> {
    let feature_map = FEATURE_CONFIG_MAP.read().unwrap();

    let mut features: Vec<_> = feature_map.iter().collect();
    features.sort_by(|(a_id, _), (b_id, _)| a_id.cmp(b_id));
    features
        .into_iter()
        .map(|(_, config)| config.clone())
        .collect()
}

/// Get the configuration for the specified adversarial feature.
fn get_adversary_feature_config(feature_id: &str) -> AdversaryFeatureConfig {
    let feature_map = FEATURE_CONFIG_MAP.read().unwrap();
    assert!(
        feature_map.contains_key(feature_id),
        "Adversarial feature {feature_id} not found in feature config map.",
    );

    feature_map.get(feature_id).unwrap().clone()
}

/// Set the configuration for the specified adversarial feature.
fn set_adversary_feature_config(feature_id: &str, config: AdversaryFeatureConfig) {
    let mut feature_map = FEATURE_CONFIG_MAP.write().unwrap();
    assert!(
        feature_map.contains_key(feature_id),
        "Adversarial feature {feature_id} not found in feature config map.",
    );

    feature_map.insert(feature_id.to_string(), config);
}
