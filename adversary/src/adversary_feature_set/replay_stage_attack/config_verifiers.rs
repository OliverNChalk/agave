//! Configuration verifiers.
//!
//! See the parent module doc-comment for an explanation as to what is it.  And also, for possible
//! ideas of how to remove this code.

use {
    super::Attack,
    crate::accounts_file::AccountsFile,
    itertools::Itertools,
    std::{
        collections::{hash_map, HashMap},
        sync::{LazyLock, Mutex},
    },
    thiserror::Error,
};

pub type AttackConfigVerifier =
    Box<dyn Fn(&AccountsFile, &Attack) -> Result<(), String> + Send + 'static>;

#[derive(Error, Debug)]
pub enum VerifierRegistrationError {
    #[error("No Attack member exists named '{0}'.")]
    UnexpectedName(&'static str),

    #[error("Entry for '{0}' is already registered at: {1}")]
    AlreadyRegistered(&'static str, &'static str),

    #[error("Verifier for '{0}' was not registered along with the rest of the verifiers.")]
    Missing(&'static str),
}

struct VerifierInfo {
    // Source location of the verifier registration or the verifier function itself.
    // To allow duplicate registration of the save verifier.
    location: &'static str,

    verifier: AttackConfigVerifier,
}

static CONFIG_VERIFIERS: LazyLock<Mutex<HashMap<&'static str, VerifierInfo>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// This is an implementation of the [`super::Attack::register_config_verifier()`].
pub(super) fn register(
    known_names: &[&str],
    name: &'static str,
    location: &'static str,
    verifier: AttackConfigVerifier,
) -> Result<(), VerifierRegistrationError> {
    if !known_names.contains(&name) {
        return Err(VerifierRegistrationError::UnexpectedName(name));
    }

    let mut verifiers = CONFIG_VERIFIERS.lock().unwrap();
    let entry = match verifiers.entry(name) {
        hash_map::Entry::Occupied(entry) => {
            let existing_location = entry.get().location;
            if location == existing_location {
                // Duplicate registration of the same verifier.
                return Ok(());
            } else {
                return Err(VerifierRegistrationError::AlreadyRegistered(
                    name,
                    existing_location,
                ));
            }
        }
        hash_map::Entry::Vacant(entry) => entry,
    };

    entry.insert(VerifierInfo { location, verifier });
    Ok(())
}

/// This is an implementation of the [`super::Attack::end_verifier_registration()`].
pub(super) fn end_registration(
    known_names: &[&'static str],
) -> Result<(), VerifierRegistrationError> {
    let verifiers = CONFIG_VERIFIERS.lock().unwrap();
    for name in known_names {
        if !verifiers.contains_key(name) {
            return Err(VerifierRegistrationError::Missing(name));
        }
    }

    Ok(())
}

/// This is an implementation of the [`super::Attack::verify()`].
pub fn verify(attack_name: &str, attack: &Attack, accounts: &AccountsFile) -> Result<(), String> {
    let verifiers = CONFIG_VERIFIERS.lock().unwrap();
    let verifier = &verifiers
        .get(attack_name)
        .unwrap_or_else(|| {
            let has_verifiers_for = if verifiers.is_empty() {
                "<none>".to_owned()
            } else {
                // When `Iterator::intersperse` is actually added to the stable part of the
                // standard library, then we can decide if we want to switch, or if we want
                // to keep using `Itertools::intersperse()`.  There seems to be no value in
                // this warning before then.
                #[allow(unstable_name_collisions)]
                verifiers
                    .keys()
                    .copied()
                    .intersperse(", ")
                    .collect::<String>()
            };

            panic!(
                "All attack verifiers should be set before the first RPC call.\nMake sure that \
                 `core::banking_stage::adversary::register_attack_config_verifiers()` has been \
                 invoked.\nMissing config verifier for: {attack_name}\nVerifiers have been \
                 registered for:\n{has_verifiers_for}"
            )
        })
        .verifier;

    verifier(accounts, attack)
}
