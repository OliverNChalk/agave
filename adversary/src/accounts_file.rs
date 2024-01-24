use {serde::Deserialize, solana_keypair::Keypair, solana_pubkey::Pubkey, std::str::FromStr};

#[derive(Default, Debug)]
pub struct AccountsFile {
    pub owner_program_id: Option<Pubkey>,
    pub payers: Vec<Keypair>,
    pub max_size: Vec<Keypair>,
    pub program_ids_jit_attack: Vec<Pubkey>,
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
            ..Default::default()
        }
    }

    pub fn with_payers_and_programs(payers_accounts: &[Keypair], program_ids: &[Pubkey]) -> Self {
        let payers = payers_accounts
            .iter()
            .map(|keypair| keypair.insecure_clone())
            .collect();
        Self {
            program_ids_jit_attack: program_ids.to_vec(),
            payers,
            ..Default::default()
        }
    }
}

impl From<AccountsFileRaw> for AccountsFile {
    fn from(raw: AccountsFileRaw) -> Self {
        let AccountsFileRaw {
            owner_program_id,
            payers,
            max_size,
            program_ids,
        } = raw;

        let payers = payers.into_iter().map(Into::into).collect();
        let max_size = max_size.into_iter().map(Into::into).collect();

        Self {
            owner_program_id: owner_program_id.map(|value| {
                Pubkey::from_str(value.as_str()).expect("Failed to parse owner program's publickey")
            }),
            payers,
            max_size,
            program_ids_jit_attack: program_ids
                .iter()
                .map(|x| Pubkey::from_str(x).expect("Failed to parse owner program's publickey"))
                .collect(),
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
    #[serde(default)]
    program_ids: Vec<String>,
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
        assert_eq!(raw.secret_key.len(), 64);
        Self::new_from_array(raw.secret_key[..32].try_into().unwrap())
    }
}
