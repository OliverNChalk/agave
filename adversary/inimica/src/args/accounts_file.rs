use {
    clap_v4::{
        self as clap,
        builder::{OsStringValueParser, TypedValueParser},
        error::{Error, ErrorKind, Result},
        Arg, Args, Command,
    },
    serde_json,
    solana_adversary::accounts_file::{AccountsFile, AccountsFileRaw},
    std::{ffi::OsStr, fs::read_to_string},
};

/// A common argument used by multiple different commands.
#[derive(Args, Debug)]
pub struct AccountsFileArgs {
    #[arg(long, value_name = "ACCOUNTS_FILE", value_parser = AccountsFileValueParser)]
    /// File that holds account addresses used by the attack.
    pub accounts: AccountsFile,
}

#[derive(Clone)]
pub struct AccountsFileValueParser;

impl TypedValueParser for AccountsFileValueParser {
    type Value = AccountsFile;

    fn parse_ref(&self, cmd: &Command, arg: Option<&Arg>, value: &OsStr) -> Result<Self::Value> {
        let path = OsStringValueParser::new().parse_ref(cmd, arg, value)?;

        let content = read_to_string(&path).map_err(|io_err| {
            Error::raw(
                ErrorKind::Io,
                format!(
                    "--accounts: Failed while reading \"{}\": {}\n",
                    path.to_string_lossy(),
                    io_err
                ),
            )
        })?;

        let accounts = serde_json::from_str::<AccountsFileRaw>(&content).map_err(|parse_err| {
            Error::raw(
                ErrorKind::Io,
                format!(
                    "--accounts: Failed while parsing \"{}\" as an accounts registry: {}\n",
                    path.to_string_lossy(),
                    parse_err
                ),
            )
        })?;

        Ok(accounts.into())
    }
}
