//! There is a number of abstractions common across different attack transaction generators.
//!
//! They should reside in this module, to help them separate from the attack transaction generators
//! themselves ([`transaction_generators`]) and the attack generator templates
//! ([`generator_templates`]).
//!
//! [`transaction_generators`]: super::transaction_generators
//! [`generator_templates`]: super::generator_templates

pub mod cycler;
pub mod index_by_modulo;

pub(crate) use {cycler::Cycler, index_by_modulo::IndexByModulo};
