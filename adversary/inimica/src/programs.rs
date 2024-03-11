//! Holds programs used by different attacks.
//!
//! TODO: `program-test` crate also holds a number of programs used in testing.  It probably makes
//! sense to extend the `program-test` API to allow access to those programs ELF, and share the
//! program library this way.

use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum KnownPrograms {
    Noop,
}

pub type ProgramElf = Vec<u8>;
pub type ProgramElfRef<'source> = &'source [u8];

pub fn program_elf(program: KnownPrograms) -> ProgramElf {
    type Library = HashMap<KnownPrograms, ProgramElf>;
    static LIBRARY: OnceLock<Mutex<Library>> = OnceLock::new();

    let library = LIBRARY.get_or_init(|| {
        Mutex::new(HashMap::from([(
            KnownPrograms::Noop,
            include_bytes!("programs/noop.so").to_vec(),
        )]))
    });
    let library = library.lock().unwrap();

    library
        .get(&program)
        .expect("Library contains ELF for all programs defined in [`KnownPrograms`]")
        .clone()
}
