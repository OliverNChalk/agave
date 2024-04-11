//! Attack on the loader v3.

use {
    crate::{
        blockhash_cache::BlockhashCache,
        deploy_program::v3::{delete_program, deploy, DeployError},
        programs::ProgramElfRef,
    },
    futures::{
        future::{BoxFuture, FutureExt as _},
        pin_mut, select,
        stream::{FuturesUnordered, StreamExt as _},
    },
    log::{debug, warn},
    parking_lot::Mutex,
    solana_clock::Slot,
    solana_hash::Hash,
    solana_instruction::Instruction,
    solana_keypair::Keypair,
    solana_metrics::metrics::MetricsSender,
    solana_pubkey::Pubkey,
    solana_rent::Rent,
    solana_rpc_client::nonblocking::rpc_client::RpcClient,
    solana_rpc_client_api::config::RpcSendTransactionConfig,
    solana_signer::Signer,
    solana_transaction::Transaction,
    std::{
        cmp::Ordering,
        collections::BinaryHeap,
        iter::Cycle,
        slice,
        sync::Arc,
        time::{Duration, Instant},
    },
    tokio::time::sleep,
};

#[derive(Debug, Clone)]
pub struct Config<'payers, 'elf> {
    pub payers: &'payers [Keypair],
    pub program: ProgramElfRef<'elf>,
    pub total_duration: Duration,
    pub execution_delay: u64,
    pub parallel_deployments: usize,
    pub max_deployed_programs: usize,
    pub min_call_iteration_duration: Duration,
    pub max_calls_per_iteration: usize,
    pub skip_program_cleanup: bool,
}

const REFRESH_BLOCKHASH_INTERVAL: Duration = Duration::from_secs(1);

pub async fn attack(
    config @ Config {
        execution_delay,
        skip_program_cleanup,
        ..
    }: Config<'_, '_>,
    _metrics: &impl MetricsSender,
    rpc_client: &RpcClient,
) -> Result<(), String> {
    let rent = Rent::default();
    let blockhash_cache = BlockhashCache::uninitialized();
    init_blockhash_cache(&blockhash_cache, rpc_client).await;

    // Programs that are being deployed.
    let mut ongoing_deployments = DeploymentsFutures::new();

    // Programs that are waiting to be called.  This is a min-heap of [`DeployedProgram`] items,
    // ordered by the call slot.  This way `call_ready()` can easily extract programs that should be
    // called next.
    let call_pending = CallPending::default();

    attack_main_loop(
        config,
        &blockhash_cache,
        &rent,
        rpc_client,
        &mut ongoing_deployments,
        call_pending.clone(),
    )
    .await?;

    if skip_program_cleanup {
        // This could leave any unfinished deployments in a broken state.
        // The expectation is that `skip_program_cleanup` will only be set for local cluster or
        // similar environments, where the blockchain state after the attack does not matter.
        //
        // If we want to set it in a longer lived environments, such as testnet, we may consider
        // waiting for all the pending deployments to finish, before we exit.
        return Ok(());
    }

    attack_cleanup(
        &blockhash_cache,
        rpc_client,
        execution_delay,
        &mut ongoing_deployments,
        call_pending,
    )
    .await;

    Ok(())
}

async fn attack_main_loop<'rpc_client, 'elf, 'deploy>(
    config @ Config {
        payers,
        total_duration,
        execution_delay,
        min_call_iteration_duration,
        max_calls_per_iteration,
        ..
    }: Config<'_, 'elf>,
    blockhash_cache: &BlockhashCache,
    rent: &Rent,
    rpc_client: &'rpc_client RpcClient,
    ongoing_deployments: &mut DeploymentsFutures<'deploy>,
    call_pending: CallPending,
) -> Result<(), String>
where
    'rpc_client: 'deploy,
    'elf: 'deploy,
{
    if payers.is_empty() {
        return Err("Need at least one payer.  Check the accounts registry.".to_owned());
    }
    let payers = payers.iter().cycle();

    let attack_end = sleep(total_duration).fuse();
    pin_mut!(attack_end);

    let refresh_blockhash_op =
        refresh_blockhash_cache(blockhash_cache, rpc_client, REFRESH_BLOCKHASH_INTERVAL).fuse();
    pin_mut!(refresh_blockhash_op);

    let deploy_and_queue_op = deploy_and_queue(
        config,
        blockhash_cache,
        rent,
        rpc_client,
        ongoing_deployments,
        call_pending.clone(),
    )
    .fuse();
    pin_mut!(deploy_and_queue_op);

    let call_ready_op = call_ready(
        blockhash_cache,
        rpc_client,
        execution_delay,
        min_call_iteration_duration,
        max_calls_per_iteration,
        payers,
        call_pending,
    )
    .fuse();
    pin_mut!(call_ready_op);

    loop {
        select! {
            _ = attack_end => break,
            _ = refresh_blockhash_op => panic!("refresh_blockhash_op() should never complete."),
            _ = deploy_and_queue_op => (),
            _ = call_ready_op => panic!("call_ready_op() should never complete."),
        }
    }

    Ok(())
}

async fn attack_cleanup(
    blockhash_cache: &BlockhashCache,
    rpc_client: &RpcClient,
    execution_delay: u64,
    ongoing_deployments: &mut DeploymentsFutures<'_>,
    call_pending: CallPending,
) {
    let mut call_pending = call_pending.lock().iter().cloned().collect::<Vec<_>>();

    let refresh_blockhash_op =
        refresh_blockhash_cache(blockhash_cache, rpc_client, REFRESH_BLOCKHASH_INTERVAL).fuse();
    pin_mut!(refresh_blockhash_op);

    let cleanup_op = async {
        loop {
            delete_all_pending(
                blockhash_cache,
                rpc_client,
                execution_delay,
                &mut call_pending,
            )
            .await;

            if ongoing_deployments.is_empty() {
                break;
            }

            if !ongoing_deployments.is_empty() {
                match ongoing_deployments.next().await {
                    Some(Ok(deployment)) => call_pending.push(deployment),
                    Some(Err(err)) => warn!("Program deployment failed: {err}"),
                    None => (),
                }
            }
        }
    }
    .fuse();
    pin_mut!(cleanup_op);

    select! {
        _ = refresh_blockhash_op => panic!("refresh_blockhash_op() should never complete."),
        _ = cleanup_op => (),
    };
}

async fn init_blockhash_cache(blockhash_cache: &BlockhashCache, rpc_client: &RpcClient) {
    loop {
        let res = blockhash_cache.refresh(rpc_client).await;
        if let Err(err) = res {
            warn!("Failed to get the latest blockhash: {err}");
        }

        // We start with not blockhash, expressed as `Hash::default()`.  We can not do anything
        // until we get at least one blockhash.
        if blockhash_cache.get() != Hash::default() {
            return;
        }
    }
}

async fn refresh_blockhash_cache(
    blockhash_cache: &BlockhashCache,
    rpc_client: &RpcClient,
    min_loop_duration: Duration,
) {
    loop {
        let loop_start = Instant::now();

        loop {
            let res = blockhash_cache.refresh(rpc_client).await;
            if let Err(err) = res {
                warn!("Failed to get the latest blockhash: {err}");
            } else {
                break;
            }
        }

        let loop_duration = loop_start.elapsed();
        if loop_duration < min_loop_duration {
            sleep(min_loop_duration.saturating_sub(loop_duration)).await;
        }
    }
}

async fn deploy_and_queue<'rpc_client, 'elf, 'deploy>(
    Config {
        payers,
        program,
        execution_delay,
        parallel_deployments,
        max_deployed_programs,
        ..
    }: Config<'_, 'elf>,
    blockhash_cache: &BlockhashCache,
    rent: &Rent,
    rpc_client: &'rpc_client RpcClient,
    ongoing_deployments: &mut DeploymentsFutures<'deploy>,
    call_pending: CallPending,
) where
    'rpc_client: 'deploy,
    'elf: 'deploy,
{
    let mut deployments_left = max_deployed_programs;
    let payers = payers.iter().cycle();

    while deployments_left > 0 {
        start_new_deployments(
            blockhash_cache,
            rent,
            rpc_client,
            execution_delay,
            parallel_deployments,
            deployments_left,
            program,
            payers.clone(),
            ongoing_deployments,
        );

        let deployed_count =
            process_deployed_programs(parallel_deployments, ongoing_deployments, &call_pending)
                .await;

        deployments_left = deployments_left.saturating_sub(deployed_count);
    }
}

// A lot of functions pass around a reference to a cycle iterator over a slice of `Keypair`s, to
// mean "payers to be used by the function".
type PayersAsSliceCycle<'keypairs> = Cycle<slice::Iter<'keypairs, Keypair>>;

/// Describes a deployed program.
///
/// A custom [`Ord`] implementation orders [`DeployedProgram`]s in reverse slot order.
/// This way a max-heap [`BinaryHeap<DeployedProgram>`] is a min-heap ordered by the slot.
#[derive(Debug)]
struct DeployedProgram {
    // Address of the program.  Private key is discarded, as it is only necessary for the initial
    // deployment.
    address: Pubkey,

    // This is the payer account used for the program deployment.  It is also the authority that can
    // be used to delete the program.
    // In general, payer and the authority could be different addresses, but in this particular
    // case, there is no reason for separating them.
    payer_and_authority: Keypair,

    call_in_slot: Slot,
}

impl PartialEq for DeployedProgram {
    fn eq(&self, other: &Self) -> bool {
        self.call_in_slot == other.call_in_slot
    }
}

impl Eq for DeployedProgram {}

impl PartialOrd for DeployedProgram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DeployedProgram {
    fn cmp(&self, other: &Self) -> Ordering {
        self.call_in_slot.cmp(&other.call_in_slot).reverse()
    }
}

impl Clone for DeployedProgram {
    fn clone(&self) -> Self {
        let DeployedProgram {
            address,
            payer_and_authority,
            call_in_slot,
        } = self;
        Self {
            address: *address,
            payer_and_authority: payer_and_authority.insecure_clone(),
            call_in_slot: *call_in_slot,
        }
    }
}

type CallPending = Arc<Mutex<BinaryHeap<DeployedProgram>>>;
type CallPendingRef<'a> = &'a Mutex<BinaryHeap<DeployedProgram>>;

type DeployFuture<'rpc_client> = BoxFuture<'rpc_client, Result<DeployedProgram, DeployError>>;

type DeploymentsFutures<'deploy> = FuturesUnordered<DeployFuture<'deploy>>;

/// Starts up to `deploy_up_to` new program deployments, but does not wait for the operations to
/// finish.  Async deploy operations are added into the `ongoing_deployments`.
///
/// Makes sure that `ongoing_deployments` size does not exceed `parallel_deployments`.
fn start_new_deployments<'rpc_client, 'elf, 'deploy>(
    blockhash_cache: &BlockhashCache,
    rent: &Rent,
    rpc_client: &'rpc_client RpcClient,
    execution_delay: u64,
    parallel_deployments: usize,
    mut deploy_up_to: usize,
    program: ProgramElfRef<'elf>,
    mut payers: PayersAsSliceCycle<'_>,
    ongoing_deployments: &mut DeploymentsFutures<'deploy>,
) where
    'rpc_client: 'deploy,
    'elf: 'deploy,
{
    while deploy_up_to > 0 && ongoing_deployments.len() < parallel_deployments {
        let program_key = Keypair::new();
        let payer = payers
            .next()
            .expect("`payers` is an infinite size non-empty iterator.")
            .insecure_clone();
        let blockhash_cache = blockhash_cache.clone();
        let rent = rent.clone();

        let address = program_key.pubkey();
        debug!("Starting deployment for {address}");

        ongoing_deployments.push(Box::pin(async move {
            deploy(
                &blockhash_cache,
                &rent,
                rpc_client,
                payer.insecure_clone(),
                program_key,
                program,
            )
            .await
            .map(|deployed_slot| DeployedProgram {
                address,
                payer_and_authority: payer,
                call_in_slot: deployed_slot.saturating_add(execution_delay),
            })
        }));

        deploy_up_to = deploy_up_to.saturating_sub(1);
    }
}

/// Adds newly deployed programs into `call_pending`.
///
/// It will process up to `parallel_deployments` of ready deployments from `ongoing_deployments`,
/// and will return.  If no deployments are ready, it will wait for at least one to be ready.
async fn process_deployed_programs(
    parallel_deployments: usize,
    ongoing_deployments: &mut DeploymentsFutures<'_>,
    call_pending: CallPendingRef<'_>,
) -> usize {
    let mut deployed = ongoing_deployments.ready_chunks(parallel_deployments);
    let Some(deployed) = deployed.next().await else {
        debug_assert!(
            false,
            "`ongoing_deployments` should never be empty, as long as we are still waiting for \
             programs to be deployed."
        );
        return 0;
    };

    let mut deployed_count: usize = 0;

    for deployment in deployed.into_iter() {
        match deployment {
            Ok(deployment) => {
                let DeployedProgram {
                    address,
                    payer_and_authority: _,
                    call_in_slot,
                } = &deployment;
                debug!("Program at {address} deployed, to be called in {call_in_slot}");

                call_pending.lock().push(deployment);
                deployed_count = deployed_count.saturating_add(1);
            }
            Err(err) => warn!("Program deployment failed: {err}"),
        }
    }

    deployed_count
}

/// Invokes up to `call_up_to` programs from `call_pending`, if they should be invoked in the
/// current slot or earlier.
///
/// As we want to run as many calls as possible, this does not wait for the call transaction to be
/// confirmed, it is just sent over the RPC and forgotten.  RPC errors that occur for the program
/// invocation are not reported to the caller, instead they are only logged.
async fn call_ready(
    blockhash_cache: &BlockhashCache,
    rpc_client: &RpcClient,
    execution_delay: u64,
    min_loop_duration: Duration,
    max_calls_per_iteration: usize,
    mut payers: PayersAsSliceCycle<'_>,
    call_pending: CallPending,
) {
    let mut current_slot = 0;
    let mut iteration_work = vec![];

    loop {
        let loop_start = Instant::now();

        call_ready_one_iteration(
            blockhash_cache,
            rpc_client,
            execution_delay,
            max_calls_per_iteration,
            &mut payers,
            &call_pending,
            &mut current_slot,
            &mut iteration_work,
        )
        .await;

        let loop_duration = loop_start.elapsed();
        if loop_duration < min_loop_duration {
            sleep(min_loop_duration.saturating_sub(loop_duration)).await;
        }
    }
}

// A single iteration of the `call_ready()`.  Extracted into a separate function, so that it can be
// unit tested.
async fn call_ready_one_iteration(
    blockhash_cache: &BlockhashCache,
    rpc_client: &RpcClient,
    execution_delay: u64,
    max_calls_per_iteration: usize,
    payers: &mut PayersAsSliceCycle<'_>,
    call_pending: CallPendingRef<'_>,
    current_slot: &mut Slot,
    iteration_work: &mut Vec<DeployedProgram>,
) {
    *current_slot = loop {
        match rpc_client.get_slot().await {
            Ok(slot) => break slot,
            Err(err) => {
                warn!("`get_slot() failed: {err}");
                // If we got our slot on the previous iteration it is OK to reuse the same slot
                // number for some time.  But if we do not know our slot at all, we need to keep
                // trying.
                if *current_slot != 0 {
                    break *current_slot;
                }
            }
        }
    };

    {
        let mut call_pending = call_pending.lock();
        while !call_pending.is_empty() && iteration_work.len() < max_calls_per_iteration {
            if call_pending.peek().unwrap().call_in_slot > *current_slot {
                break;
            }
            iteration_work.push(call_pending.pop().unwrap());
        }
    }

    let blockhash = blockhash_cache.get();

    for DeployedProgram {
        address,
        call_in_slot,
        ..
    } in iteration_work.iter_mut()
    {
        let call_program = Instruction::new_with_bytes(*address, &[], vec![]);
        let payer = payers
            .next()
            .expect("`payers` is an infinite size non-empty iterator.");
        let transaction = Transaction::new_signed_with_payer(
            &[call_program],
            Some(&payer.pubkey()),
            &[payer],
            blockhash,
        );

        match rpc_client
            .send_transaction_with_config(
                &transaction,
                RpcSendTransactionConfig {
                    skip_preflight: true,
                    preflight_commitment: None,
                    ..RpcSendTransactionConfig::default()
                },
            )
            .await
        {
            Err(err) => {
                warn!("Slot {current_slot}: Program at {address}: send_transaction() failed: {err}",)
            }
            Ok(signature) => {
                debug!("Slot {current_slot}: Invoked program at {address}: {signature}",)
            }
        }

        *call_in_slot = current_slot.saturating_add(execution_delay);
    }

    {
        let mut call_pending = call_pending.lock();
        call_pending.extend(iteration_work.drain(..));
    }
}

/// Deletes accounts for programs in `call_pending`, and removes those programs from `call_pending`.
///
/// A program can not be deleted in the slot in which it was deployed.  So, if any of the programs
/// have been deployed in the current slot, this function will wait for the slot to change, to be
/// able to delete everything.
async fn delete_all_pending(
    blockhash_cache: &BlockhashCache,
    rpc_client: &RpcClient,
    execution_delay: u64,
    call_pending: &mut Vec<DeployedProgram>,
) {
    let mut current_slot = loop {
        match rpc_client.get_slot().await {
            Ok(slot) => break slot,
            Err(err) => warn!("Failed to get the current slot: {err}"),
        }
    };

    loop {
        // This is a bit of a hack.  We deduct a conservative estimate of the deployment slot based on
        // the next call slot.  Program was deployed at least `execution_delay` slots before the
        // `call_in_slot` set for it.
        let can_be_deleted = |deployment: &DeployedProgram| {
            deployment.call_in_slot.saturating_sub(execution_delay) < current_slot
        };

        for DeployedProgram {
            address,
            payer_and_authority,
            ..
        } in call_pending
            .iter()
            .filter(|deployment| can_be_deleted(deployment))
        {
            let res =
                delete_program(blockhash_cache, rpc_client, payer_and_authority, address).await;

            if let Err(err) = res {
                warn!("Failed to delete program at {address}: {err}");
            }
        }

        call_pending.retain(|deployment| !can_be_deleted(deployment));

        // Make sure `delete_all_pending()` is not invoked too frequently.
        if call_pending.is_empty() {
            return;
        }

        // Wait for the slot to change.
        loop {
            match rpc_client.get_slot().await {
                Ok(slot) if slot != current_slot => {
                    current_slot = slot;
                    break;
                }
                Ok(_) => {
                    // Do not loop too fast, even if our RPC is fast.
                    sleep(Duration::from_millis(100)).await;
                }
                Err(err) => warn!("Failed to get the current slot: {err}"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use {
        super::{
            attack, call_ready_one_iteration, deploy_and_queue, init_blockhash_cache,
            process_deployed_programs, start_new_deployments, CallPending, CallPendingRef, Config,
            DeployedProgram, DeploymentsFutures,
        },
        crate::{
            blockhash_cache::BlockhashCache,
            programs::{program_elf, KnownPrograms},
        },
        futures::{future::FutureExt as _, pin_mut, select},
        humantime::Duration as HumanDuration,
        log::debug,
        solana_hash::Hash,
        solana_metrics::metrics::test_mocks::MockMetricsSender,
        solana_pubkey::Pubkey,
        solana_rent::Rent,
        solana_test_validator::TestValidatorGenesis,
        std::{collections::HashMap, time::Duration},
        tokio::{self, time::sleep},
    };

    // Could not figure out how to prepare the common setup using a function.  Values are moved and
    // references used by some of the common values do not work well together.
    //
    // In particular, `ongoing_deployments` holds a reference to `rpc_client`, and the compiler is
    // unhappy, when I create an `rpc_client` via the `test_validator.get_async_rpc_client()` call,
    // thinking that the produced `rpc_client` does not live long enough, as it is referenced by the
    // `ongoing_deployments`.  Even while `ongoing_deployments`.  This fails even if I specify the
    // drop order manually, via explicit `drop()` calls.
    macro_rules! common_setup {
        (
            $test_validator:ident,
            $payer:ident,
            $rpc_client:ident,
            $blockhash_cache:ident,
            $rent:ident,
            $elf:ident,
            mut $ongoing_deployments:ident $(,)?
        ) => {
            let ($test_validator, $payer) = TestValidatorGenesis::default().start_async().await;
            let $rpc_client = $test_validator.get_async_rpc_client();
            let $blockhash_cache = BlockhashCache::uninitialized();
            let $rent = Rent::default();

            let $elf = program_elf(KnownPrograms::Noop);

            let mut $ongoing_deployments = DeploymentsFutures::new();
        };
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn simple_init_blockhash_cache() {
        common_setup! {
            test_validator,
            _payer,
            rpc_client,
            blockhash_cache,
            _rent,
            _elf,
            mut _ongoing_deployments,
        };

        init_blockhash_cache(&blockhash_cache, &rpc_client).await;
        assert_ne!(blockhash_cache.get(), Hash::default());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn simple_start_new_deployemnt() {
        common_setup! {
            test_validator,
            payer,
            rpc_client,
            blockhash_cache,
            rent,
            elf,
            mut ongoing_deployments,
        };
        let payers = &[payer];
        let payers = payers.iter().cycle();

        let execution_delay = 100;
        let parallel_deployments = 10;
        let deploy_up_to = 5;
        assert!(
            parallel_deployments > deploy_up_to,
            "First check assumption below"
        );

        init_blockhash_cache(&blockhash_cache, &rpc_client).await;

        start_new_deployments(
            &blockhash_cache,
            &rent,
            &rpc_client,
            execution_delay,
            parallel_deployments,
            deploy_up_to,
            &elf,
            payers.clone(),
            &mut ongoing_deployments,
        );

        assert_eq!(
            ongoing_deployments.len(),
            deploy_up_to,
            "start_new_deployments() should not deploy more than `deploy_up_to` programs, even if \
             `parallel_deployments` allows more.\nongoing_deployments.len(): \
             {}\nparallel_deployments: {}\ndeploy_up_to: {}",
            ongoing_deployments.len(),
            parallel_deployments,
            deploy_up_to,
        );

        let deploy_up_to = parallel_deployments.saturating_mul(2);

        start_new_deployments(
            &blockhash_cache,
            &rent,
            &rpc_client,
            execution_delay,
            parallel_deployments,
            deploy_up_to,
            &elf,
            payers,
            &mut ongoing_deployments,
        );

        assert_eq!(
            ongoing_deployments.len(),
            parallel_deployments,
            "start_new_deployments() should not start more than `parallel_deployments` \
             deployments at the same time, even if `deploy_up_to` allows \
             more.\nongoing_deployments.len(): {}\nparallel_deployments: {}\ndeploy_up_to: {}",
            ongoing_deployments.len(),
            parallel_deployments,
            deploy_up_to,
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn simple_process_deployed_programs() {
        common_setup! {
            test_validator,
            payer,
            rpc_client,
            blockhash_cache,
            rent,
            elf,
            mut ongoing_deployments,
        };
        let payers = &[payer];
        let payers = payers.iter().cycle();

        let execution_delay = 100;
        let parallel_deployments = 10;
        let deploy_up_to = 5;

        init_blockhash_cache(&blockhash_cache, &rpc_client).await;

        start_new_deployments(
            &blockhash_cache,
            &rent,
            &rpc_client,
            execution_delay,
            parallel_deployments,
            deploy_up_to,
            &elf,
            payers,
            &mut ongoing_deployments,
        );

        assert_eq!(ongoing_deployments.len(), deploy_up_to);

        let call_pending = CallPending::default();

        // It takes about 50 seconds to deploy a single program :(
        // The only upside is that all programs are deployed in parallel, so it takes 50 seconds to
        // deploy 5 programs as well.
        let max_runtime = Duration::from_secs(90);
        let timeout = sleep(max_runtime).fuse();
        pin_mut!(timeout);

        loop {
            let process_deployed_programs_op = process_deployed_programs(
                parallel_deployments,
                &mut ongoing_deployments,
                &call_pending,
            )
            .fuse();
            pin_mut!(process_deployed_programs_op);

            select! {
                _ = timeout => panic!(
                    "Failed to deploy {} programs in {}",
                    deploy_up_to,
                    HumanDuration::from(max_runtime),
                ),
                _deployed_count = process_deployed_programs_op =>
                {
                    let ready = call_pending.lock().len();
                    if ready == deploy_up_to {
                        break;
                    }
                }
            }

            sleep(Duration::from_millis(100)).await;
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn simple_deploy_and_queue() {
        common_setup! {
            test_validator,
            payer,
            rpc_client,
            blockhash_cache,
            rent,
            elf,
            mut ongoing_deployments,
        };
        let payers = &[payer];

        let parallel_deployments = 5;
        let max_deployed_programs = 10;

        init_blockhash_cache(&blockhash_cache, &rpc_client).await;

        let call_pending = CallPending::default();

        // See `simple_process_deployed_programs()` for the timeout note.
        let max_runtime = Duration::from_secs(60 * 2 + 30);
        let timeout = sleep(max_runtime).fuse();
        pin_mut!(timeout);

        let deploy_and_queue_op = deploy_and_queue(
            Config {
                payers,
                program: &elf,
                total_duration: Duration::from_millis(0),
                execution_delay: 0,
                parallel_deployments,
                max_deployed_programs,
                min_call_iteration_duration: Duration::from_millis(0),
                max_calls_per_iteration: 0,
                skip_program_cleanup: false,
            },
            &blockhash_cache,
            &rent,
            &rpc_client,
            &mut ongoing_deployments,
            call_pending.clone(),
        )
        .fuse();
        pin_mut!(deploy_and_queue_op);

        select! {
            _ = timeout => panic!(
                "Failed to deploy {} programs in {}",
                max_deployed_programs,
                HumanDuration::from(max_runtime),
            ),
            _ = deploy_and_queue_op => (),
        };

        {
            let call_pending = call_pending.lock();
            assert_eq!(call_pending.len(), max_deployed_programs);
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn simple_call_ready_one_iteration() {
        common_setup! {
            test_validator,
            payer,
            rpc_client,
            blockhash_cache,
            rent,
            elf,
            mut ongoing_deployments,
        };
        let payers = &[payer];

        let parallel_deployments = 10;
        let max_deployed_programs = 10;

        init_blockhash_cache(&blockhash_cache, &rpc_client).await;

        let call_pending = CallPending::default();

        // === Deploy programs, before we can call them.

        {
            // See `simple_process_deployed_programs()` for the timeout note.
            let max_runtime = Duration::from_secs(60);
            let timeout = sleep(max_runtime).fuse();
            pin_mut!(timeout);

            let deploy_and_queue_op = deploy_and_queue(
                Config {
                    payers,
                    program: &elf,
                    total_duration: Duration::from_millis(0),
                    execution_delay: 0,
                    parallel_deployments,
                    max_deployed_programs,
                    min_call_iteration_duration: Duration::from_millis(0),
                    max_calls_per_iteration: 0,
                    skip_program_cleanup: false,
                },
                &blockhash_cache,
                &rent,
                &rpc_client,
                &mut ongoing_deployments,
                call_pending.clone(),
            )
            .fuse();
            pin_mut!(deploy_and_queue_op);

            select! {
                _ = timeout => panic!(
                    "Failed to deploy {} programs in {}",
                    max_deployed_programs,
                    HumanDuration::from(max_runtime),
                ),
                _ = deploy_and_queue_op => (),
            };
        }

        {
            let call_pending = call_pending.lock();
            assert_eq!(call_pending.len(), max_deployed_programs);
        }

        debug!("Done with deployments.");

        // === Call.
        //
        // `execution_delay` is set to `0` in the `deploy_and_queue()` call, so all programs should
        // be ready to be invoked.

        let execution_delay = 100;

        fn all_call_in_slots<SetCallInSlot: Fn(u64) -> u64>(
            call_pending: CallPendingRef<'_>,
            set_call_in_slot: SetCallInSlot,
        ) -> HashMap<Pubkey, u64> {
            let call_pending = call_pending.lock();
            call_pending
                .iter()
                .map(
                    |DeployedProgram {
                         address,
                         call_in_slot,
                         ..
                     }| { (*address, set_call_in_slot(*call_in_slot)) },
                )
                .collect::<HashMap<_, _>>()
        }

        let mut expected_call_in_slot = all_call_in_slots(&call_pending, |_| execution_delay);

        let mut current_slot = 0;
        let mut iteration_work = vec![];

        let mut payers = payers.iter().cycle();

        {
            let max_runtime = Duration::from_secs(5);
            let timeout = sleep(max_runtime).fuse();
            pin_mut!(timeout);

            let call_ready_one_iteration_op = call_ready_one_iteration(
                &blockhash_cache,
                &rpc_client,
                execution_delay,
                usize::MAX,
                &mut payers,
                &call_pending,
                &mut current_slot,
                &mut iteration_work,
            )
            .fuse();
            pin_mut!(call_ready_one_iteration_op);

            select! {
                _ = timeout => panic!(
                    "Failed to call ready programs in {}",
                    HumanDuration::from(max_runtime),
                ),
                _ = call_ready_one_iteration_op => (),
            };
        }

        debug!("Done with calls.");

        expected_call_in_slot
            .values_mut()
            .for_each(|v| *v = current_slot.saturating_add(execution_delay));
        let actual_call_in_slot = all_call_in_slots(&call_pending, |v| v);

        assert_ne!(current_slot, 0);
        assert_eq!(iteration_work, vec![]);

        assert_eq!(expected_call_in_slot, actual_call_in_slot);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn simple_attack() {
        common_setup! {
            test_validator,
            payer,
            rpc_client,
            _blockhash_cache,
            _rent,
            elf,
            mut _ongoing_deployments,
        };
        let payers = &[payer];

        let max_runtime = Duration::from_secs(120);
        let timeout = sleep(max_runtime).fuse();
        pin_mut!(timeout);

        let metrics_sender = MockMetricsSender::default();

        let attack_op = attack(
            Config {
                payers,
                program: &elf,
                total_duration: Duration::from_secs(45),
                execution_delay: 10,
                parallel_deployments: 2,
                max_deployed_programs: 20,
                min_call_iteration_duration: Duration::from_millis(100),
                max_calls_per_iteration: 10,
                skip_program_cleanup: false,
            },
            &metrics_sender,
            &rpc_client,
        )
        .fuse();
        pin_mut!(attack_op);

        debug!("Starting attack.");

        select! {
            _ = timeout => panic!(
                "Failed to finish the attack in {}",
                HumanDuration::from(max_runtime),
            ),
            res = attack_op => res.expect("All attack parameters are correct"),
        };

        debug!("Attack done.");
    }
}
