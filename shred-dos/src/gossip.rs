use {
    crate::ShredDosError,
    solana_connection_cache::connection_cache::Protocol,
    solana_gossip::gossip_service::discover,
    solana_keypair::Keypair,
    solana_pubkey::Pubkey,
    solana_rpc_client::rpc_client::RpcClient,
    solana_streamer::socket::SocketAddrSpace,
    std::{net::SocketAddr, str::FromStr, time::Duration},
};

pub(crate) fn get_top_staked_tvu_addrs(
    identity_keypair: Keypair,
    json_rpc_url: &str,
    gossip_entrypoint: &SocketAddr,
    num_nodes: usize,
    allow_private_addr: bool,
) -> Result<Vec<SocketAddr>, ShredDosError> {
    let rpc_client = RpcClient::new(json_rpc_url);
    let mut vote_accounts = rpc_client.get_vote_accounts()?;
    let num_validators = vote_accounts
        .current
        .len()
        .saturating_add(vote_accounts.delinquent.len());
    vote_accounts
        .current
        .sort_unstable_by_key(|k| k.activated_stake);
    let top_staked_vote_accounts = vote_accounts
        .current
        .iter()
        .rev()
        .take(num_nodes)
        .collect::<Vec<_>>();
    let top_staked_pubkeys = top_staked_vote_accounts
        .iter()
        .map(|a| Pubkey::from_str(&a.node_pubkey).unwrap())
        .collect::<Vec<_>>();

    let discover_timeout = Duration::from_secs(u64::MAX);
    let my_shred_version = 0;
    let socket_addr_space = SocketAddrSpace::new(allow_private_addr);
    let (_, validators) = discover(
        Some(identity_keypair),
        Some(gossip_entrypoint),
        Some(num_validators),
        discover_timeout,
        Some(&top_staked_pubkeys),
        None, // find_node_by_gossip_addr
        None, // my_gossip_addr
        my_shred_version,
        socket_addr_space,
    )?;

    let tvu_socket_addrs = top_staked_pubkeys
        .iter()
        .map(|id| {
            validators
                .iter()
                .find(|v| v.pubkey() == id)
                .expect("Could not spy node {id}")
                .tvu(Protocol::UDP)
                .expect("Could not get a UDP tvu forward address")
        })
        .collect::<Vec<_>>();

    Ok(tvu_socket_addrs)
}
