use {
    solana_program_test::{ProgramTest, ProgramTestContext},
    solana_sdk::{
        account::Account,
        clock::Epoch,
        pubkey::Pubkey,
        signature::Keypair,
        signer::Signer,
        vote::state::{VoteState, VoteStateVersions},
    },
};

#[tokio::test]
async fn repro() {
    let program_test = ProgramTest::default();
    let mut context = program_test.start_with_context().await;
    let mut slot = context.banks_client.get_root_slot().await.unwrap();

    // Create a vote account.
    let authority = Keypair::new();
    let validator = Pubkey::new_unique();
    create_vote_account(&mut context, &validator, &authority.pubkey()).await;

    // Jump 1 slot, works.
    slot += 1;
    context.warp_to_slot(slot).unwrap();

    // Jump 1 slot, works again.
    slot += 1;
    context.warp_to_slot(slot).unwrap();

    // Jump 2 slots, panics.
    slot += 2;
    context.warp_to_slot(slot).unwrap();
}

async fn create_vote_account(context: &mut ProgramTestContext, node: &Pubkey, authority: &Pubkey) {
    let vote = Pubkey::new_unique();
    let root_slot = context.banks_client.get_root_slot().await.unwrap();

    let mut vote_state = VoteState::new_rand_for_tests(*node, root_slot);
    vote_state.authorized_withdrawer = *authority;

    let mut data = vec![0; VoteState::size_of()];
    VoteState::serialize(&VoteStateVersions::new_current(vote_state), &mut data).unwrap();

    let vote_account = Account {
        lamports: 5_000_000_000, // 5 SOL
        data,
        owner: solana_sdk::vote::program::ID,
        executable: false,
        rent_epoch: Epoch::default(),
    };

    context.set_account(&vote, &vote_account.into());
}
