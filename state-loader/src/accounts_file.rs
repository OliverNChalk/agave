//! Structure used for serializing and deserializing created accounts.
use {
    crate::{
        accounts_creator::AccountsCreator,
        cli::{AccountParams, WriteAccounts},
        error::StateLoaderError,
        validate_accounts::validate,
    },
    serde::{Deserialize, Serialize},
    solana_client::nonblocking::rpc_client::RpcClient,
    solana_keypair::Keypair,
    solana_pubkey::Pubkey,
    solana_signer::Signer,
    std::{fs::File, io::Write, path::PathBuf, str::FromStr, sync::Arc},
};

#[derive(Default, Debug, PartialEq)]
pub struct AccountsFile {
    /// Account owner for max_size and sized_accounts.
    /// If specified, program can modify these accounts.
    pub owner_program_id: Pubkey,
    /// Accounts used to pay for transactions.
    /// Many are used to avoid introducing dependencies between transactions.
    pub payers: Vec<Keypair>,
    /// Accounts of arbitrary size.
    pub sized_accounts: Vec<Keypair>,
}

pub fn read_accounts_file(path: PathBuf) -> AccountsFile {
    let file_content = std::fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!("Failed to read the accounts file.\nPath: {path:?}\nError: {err}")
    });
    serde_json::from_str::<AccountsFileRaw>(&file_content)
        .unwrap_or_else(|err| {
            panic!(
                "Failed to parse accounts file.\nPath: {path:?}\nError: \
                 {err}\nContent:\n{file_content}"
            )
        })
        .into()
}

pub fn write_accounts_file(path: PathBuf, accounts: AccountsFile) {
    let accounts_file_raw: AccountsFileRaw = accounts.into();
    let file_content = serde_json::to_string(&accounts_file_raw)
        .unwrap_or_else(|err| panic!("Failed to serialize the accounts file.\nError: {err}"));
    let mut file = File::create(path.clone())
        .unwrap_or_else(|err| panic!("Failed to create a file.\nPath: {path:?}\nError: {err}"));

    file.write_all(file_content.as_bytes())
        .unwrap_or_else(|err| {
            panic!(
                "Failed to write the accounts file.\nPath: {path:?}\nError: \
                 {err}\nContent:\n{file_content}"
            )
        });
}

pub async fn create_ephemeral_accounts(
    rpc_client: Arc<RpcClient>,
    authority: Keypair,
    account_params: AccountParams,
    validate_accounts: bool,
) -> Result<AccountsFile, StateLoaderError> {
    let accounts_creator = AccountsCreator::new(rpc_client.clone(), authority, account_params);
    let accounts = accounts_creator.create().await?;
    if validate_accounts && !validate(&accounts, rpc_client, account_params).await? {
        return Err(StateLoaderError::AccountsValidationFailure);
    }

    Ok(accounts)
}

pub async fn create_file_persisted_accounts(
    rpc_client: Arc<RpcClient>,
    authority: solana_keypair::Keypair,
    write_accounts: WriteAccounts,
    validate_accounts: bool,
) -> Result<(), crate::error::StateLoaderError> {
    let accounts = create_ephemeral_accounts(
        rpc_client,
        authority,
        write_accounts.account_params,
        validate_accounts,
    )
    .await?;

    write_accounts_file(write_accounts.accounts_file, accounts);

    Ok(())
}

impl From<AccountsFileRaw> for AccountsFile {
    fn from(
        AccountsFileRaw {
            owner_program_id,
            payers,
            sized_accounts,
        }: AccountsFileRaw,
    ) -> Self {
        let payers = payers.into_iter().map(Into::into).collect();
        let sized_accounts = sized_accounts.into_iter().map(Into::into).collect();

        Self {
            owner_program_id: Pubkey::from_str(owner_program_id.as_str())
                .expect("Failed to parse owner program's publickey"),
            payers,
            sized_accounts,
        }
    }
}

#[derive(Deserialize, Serialize)]
struct AccountsFileRaw {
    #[serde(default)]
    owner_program_id: String,
    #[serde(default)]
    payers: Vec<KeypairRaw>,
    #[serde(default)]
    sized_accounts: Vec<KeypairRaw>,
}

impl From<AccountsFile> for AccountsFileRaw {
    fn from(
        AccountsFile {
            owner_program_id,
            payers,
            sized_accounts,
        }: AccountsFile,
    ) -> Self {
        AccountsFileRaw {
            owner_program_id: owner_program_id.to_string(),
            payers: payers.iter().map(KeypairRaw::from).collect(),
            sized_accounts: sized_accounts.iter().map(KeypairRaw::from).collect(),
        }
    }
}

#[derive(Deserialize, Serialize)]
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

impl From<&Keypair> for KeypairRaw {
    fn from(keypair: &Keypair) -> Self {
        Self {
            _pubkey: keypair.pubkey().to_string(),
            secret_key: keypair.to_bytes().to_vec(),
        }
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
            "sized_accounts": [
                {
                    "publicKey": pubkey,
                    "secretKey": secretkey,
                },
            ]
        })
        .to_string();

        let accounts = serde_json::from_str::<AccountsFileRaw>(&json_data)
            .expect("The test json should be properly formatted.");
        let actual = AccountsFile::from(accounts);
        assert_eq!(
            actual,
            AccountsFile {
                owner_program_id: keypair.pubkey(),
                payers: vec![keypair.insecure_clone(), keypair.insecure_clone()],
                sized_accounts: vec![keypair],
            }
        );
    }

    #[test]
    fn test_serialize_full_accounts_file() {
        let keypair = Keypair::new();
        let pubkey = keypair.pubkey().to_string();
        let secretkey = format!("{:?}", keypair.to_bytes());
        let expected = AccountsFile {
            owner_program_id: keypair.pubkey(),
            payers: vec![keypair.insecure_clone(), keypair.insecure_clone()],
            sized_accounts: vec![keypair],
        };

        let accounts_file_raw: AccountsFileRaw = expected.into();
        let actual_json_data = serde_json::to_string(&accounts_file_raw).unwrap();

        // we cannot use json! macro because it doesn't guarantee the order of fields.
        let mut json_data = format!(
            r#"
        {{
            "owner_program_id": "{pubkey}",
            "payers": [
                {{
                    "publicKey": "{pubkey}",
                    "secretKey": {secretkey}
                }},
                {{
                    "publicKey": "{pubkey}",
                    "secretKey": {secretkey}
                }}
            ],
            "sized_accounts": [
                {{
                    "publicKey": "{pubkey}",
                    "secretKey": {secretkey}
                }}
            ]
        }}
        "#
        );
        json_data.retain(|c| !c.is_ascii_whitespace());

        assert_eq!(actual_json_data, json_data);
    }
}
