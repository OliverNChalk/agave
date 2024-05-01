mod adversary;
mod args;
mod common;

fn main() -> Result<(), String> {
    solana_logger::setup();
    args::run_command()
}
