mod adversary;
mod args;
mod continuous_mode;

fn main() -> Result<(), String> {
    solana_logger::setup();
    args::run_command()
}
