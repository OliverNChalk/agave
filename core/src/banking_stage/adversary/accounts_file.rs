use {
    enumset::EnumSetType,
    serde::Deserialize,
    solana_keypair::Keypair,
    solana_pubkey::Pubkey,
    std::str::FromStr,
    strum::VariantNames,
    strum_macros::{EnumString, EnumVariantNames},
};

#[derive(Debug, EnumSetType, EnumString, EnumVariantNames)]
#[strum(serialize_all = "kebab-case")]
pub enum BlockGeneratorOption {
    TransferRandom,
    CreateNonceAccounts,
    AllocateRandomLarge,
    AllocateRandomSmall,
    ChainTransactions,
    WriteProgram,
}

impl BlockGeneratorOption {
    pub const fn cli_names() -> &'static [&'static str] {
        Self::VARIANTS
    }

    pub fn cli_message() -> &'static str {
        "Specify type of the attack, if not specified all attacks except for program-based will be \
         launched in a round-robin fashion"
    }
}

#[derive(Default, Debug)]
pub struct AccountsFile {
    pub owner_program_id: Option<Pubkey>,
    pub payers: Vec<Keypair>,
    pub max_size: Vec<Keypair>,
}

impl AccountsFile {
    pub fn with_payers(payers: &[Keypair]) -> Self {
        let payers = payers
            .iter()
            .map(|keypair| keypair.insecure_clone())
            .collect();
        Self {
            payers,
            ..Default::default()
        }
    }
    pub fn with_payers_and_max_size(
        owner_program_id: &Pubkey,
        payers_accounts: &[Keypair],
        max_size_accounts: &[Keypair],
    ) -> Self {
        let payers = payers_accounts
            .iter()
            .map(|keypair| keypair.insecure_clone())
            .collect();
        let max_size = max_size_accounts
            .iter()
            .map(|keypair| keypair.insecure_clone())
            .collect();
        Self {
            owner_program_id: Some(*owner_program_id),
            payers,
            max_size,
        }
    }
}

impl From<AccountsFileRaw> for AccountsFile {
    fn from(raw: AccountsFileRaw) -> Self {
        let AccountsFileRaw {
            owner_program_id,
            payers,
            max_size,
        } = raw;

        let payers = payers.into_iter().map(Into::into).collect();
        let max_size = max_size.into_iter().map(Into::into).collect();

        Self {
            owner_program_id: owner_program_id.map(|value| {
                Pubkey::from_str(value.as_str()).expect("Failed to parse owner program's publickey")
            }),
            payers,
            max_size,
        }
    }
}

#[derive(Deserialize)]
pub struct AccountsFileRaw {
    #[serde(default)]
    owner_program_id: Option<String>,
    #[serde(default)]
    payers: Vec<KeypairRaw>,
    #[serde(default)]
    max_size: Vec<KeypairRaw>,
}

#[derive(Deserialize)]
struct KeypairRaw {
    #[serde(rename = "publicKey")]
    pub _pubkey: String,
    #[serde(rename = "secretKey")]
    pub secret_key: Vec<u8>,
}

impl From<KeypairRaw> for Keypair {
    fn from(raw: KeypairRaw) -> Self {
        Self::new_from_array(raw.secret_key.try_into().unwrap())
    }
}
