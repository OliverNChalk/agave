//! Some attacks are quite similar.  Common parts of those are extracted into reusable pieces in
//! this module.
//!
//! We already have a lot of different attacks.  Separation helps with the code organization.
//!
//! If a certain attack is complex enough to have an internal structure, it should be described as a
//! module with its own code under `template_generators`.  Similarly, a complex attack template
//! might consist of a module of its own.

pub mod max_accounts_tx;
pub mod rotate_accounts;
