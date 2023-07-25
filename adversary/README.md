# Solana Adversarial Features

The goal of this crate is to offer a framework to configure, check status,
and enable/disable adversarial features in a uniform matter. This provided
framework is then exposed for configuration via an RPC interface. This has
several advantages, such as allowing on-the-fly adjustments of the invalidator
and configuring multiple invalidators with simple commands.

## Adding a new feature
The following steps outline how to go about adding a new adversarial feature.
For the sake of illustration, these steps will introduce a new adversarial
feature called `example`.

1. Add a new module to `adversary/src/adversary_feature_set.rs`. The
   module should contain a unique string ID and an AdversarialConfig that
   contains any configurable parameters for the new feature.

```rust
pub mod example {
    pub const ID: &str = "example";
    adversarial_feature_impl!(Example);

    #[derive(Clone, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct AdversarialConfig {
        pub example_num: u64,
    }
}
```

2. Add an enum variant for the new feature to `AdversaryFeatureConfig`.
```rust
    #[serde(rename = "exampleAdversarialConfig")]
    Example(example::AdversarialConfig),
```

3. Add an entry for the ID / configuration struct to the feature map.
```rust
            (
                example::ID,
                AdversaryFeatureConfig::Example(example::AdversarialConfig::default())
            ),
```

4. Add a new method to the `Adversary` rpc `trait` in `rpc/src/rpc_adversary.rs`.
```rust
    #[rpc(meta, name = "configureExample")]
    fn configure_example(
        &self,
        meta: Self::Metadata,
        config: example::AdversarialConfig,
    ) -> Result<()>;
```

5. Implement the new method in `AdversaryImpl` in the same file as above. The
   implementation should be minimal as it just needs to update the feature map.
```rust
    fn configure_example(
        &self,
        _meta: Self::Metadata,
        config: example::AdversarialConfig,
    ) -> Result<()> {
        self.perform_configuration(meta, || Ok(example::set_config(config)))
    }
```

6. To write which attack has been launched to the metrics database, extend function `output_adversary_metrics`.

7. In the desired location, use code similar to below to access configuration
   for the desired adversarial feature.
```rust
// ...
let config = adversary_feature_set::example::get_config();
// ... some code that uses this config
```

