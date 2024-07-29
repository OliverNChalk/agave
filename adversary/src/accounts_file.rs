use {serde::Deserialize, solana_keypair::Keypair, solana_pubkey::Pubkey, std::str::FromStr};

#[derive(Default, Debug, PartialEq)]
pub struct AccountsFile {
    /// Account owner for max_size and sized_accounts.
    /// If specified, program can modify these accounts.
    pub owner_program_id: Option<Pubkey>,
    /// Accounts used to pay for transactions.
    /// Many are used to avoid introducing dependencies between transactions.
    pub payers: Vec<Keypair>,
    /// Accounts of size `MAX_PERMITTED_DATA_LENGTH`
    pub max_size: Vec<Keypair>,
    /// Pubkeys of programs used for program cache attacks.
    pub program_ids_jit_attack: Vec<Pubkey>,
}

impl AccountsFile {
    pub fn new(
        owner_program_id: Option<Pubkey>,
        payers: Option<&[Keypair]>,
        max_size: Option<&[Keypair]>,
        program_ids_jit_attack: Option<&[Pubkey]>,
    ) -> Self {
        let payers: Vec<Keypair> = clone_accounts_or_empty(payers);
        let max_size: Vec<Keypair> = clone_accounts_or_empty(max_size);
        let program_ids_jit_attack: Vec<Pubkey> = match program_ids_jit_attack {
            Some(program_ids_jit_attack) => program_ids_jit_attack.to_vec(),
            None => Vec::new(),
        };

        Self {
            owner_program_id,
            payers,
            max_size,
            program_ids_jit_attack,
        }
    }
}

impl Clone for AccountsFile {
    fn clone(&self) -> Self {
        let Self {
            owner_program_id,
            payers,
            max_size,
            program_ids_jit_attack,
        } = self;

        Self {
            owner_program_id: *owner_program_id,
            payers: payers.iter().map(Keypair::insecure_clone).collect(),
            max_size: max_size.iter().map(Keypair::insecure_clone).collect(),
            program_ids_jit_attack: program_ids_jit_attack.clone(),
        }
    }
}

fn clone_accounts_or_empty(accounts: Option<&[Keypair]>) -> Vec<Keypair> {
    match accounts {
        Some(accounts) => accounts.iter().map(Keypair::insecure_clone).collect(),
        None => Vec::new(),
    }
}

impl From<AccountsFileRaw> for AccountsFile {
    fn from(
        AccountsFileRaw {
            owner_program_id,
            payers,
            max_size,
            program_ids,
        }: AccountsFileRaw,
    ) -> Self {
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

#[cfg(test)]
mod tests {
    use {super::*, solana_signer::Signer};

    #[test]
    fn test_deserialize_full_accounts_file() {
        let keypair = Keypair::new();
        let pubkey = keypair.pubkey().to_string();
        let secretkey: Vec<u8> = keypair.to_bytes().to_vec();
        let json_data = serde_json::json!({
            "owner_program_id": pubkey,
            "payers": [
                {
                    "publicKey": pubkey,
                    "secretKey": secretkey,
                },
                {
                    "publicKey": pubkey,
                    "secretKey": secretkey,
                },
            ],
            "max_size": [
                {
                    "publicKey": pubkey,
                    "secretKey": secretkey,
                },
            ],
            "program_ids": [pubkey, pubkey],
        })
        .to_string();

        let accounts = serde_json::from_str::<AccountsFileRaw>(&json_data)
            .expect("The test json should be properly formatted.");
        assert_eq!(
            AccountsFile::from(accounts),
            AccountsFile {
                owner_program_id: Some(keypair.pubkey()),
                payers: vec![keypair.insecure_clone(), keypair.insecure_clone()],
                max_size: vec![keypair.insecure_clone()],
                program_ids_jit_attack: vec![keypair.pubkey(), keypair.pubkey()],
            }
        )
    }
}
