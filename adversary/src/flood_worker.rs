use {
    rayon::{prelude::*, ThreadPool},
    solana_keypair::Keypair,
    solana_rayon_threadlimit::get_thread_count,
    std::{
        sync::{Arc, Condvar, Mutex, RwLock},
        thread::{self, JoinHandle},
        time::Duration,
    },
};

#[derive(Debug, Default)]
pub struct ExitCondition {
    set: Mutex<bool>,
    wait_condition: Condvar,
}

impl ExitCondition {
    fn set(&self) {
        *self.set.lock().unwrap() = true;
        self.wait_condition.notify_all();
    }

    pub fn is_set(&self) -> bool {
        *self.set.lock().unwrap()
    }

    pub fn wait_is_set(&self, max_wait: Duration) -> bool {
        let set = self.set.lock().unwrap();
        if *set {
            return true;
        }
        let (set, _wait_timeout_result) = self
            .wait_condition
            .wait_timeout_while(set, max_wait, |set_val| !(*set_val))
            .unwrap();
        *set
    }
}

#[derive(Debug)]
pub struct AdversaryWorkersContext {
    exit: Arc<ExitCondition>,
    thread_hdls: Vec<JoinHandle<()>>,
}

impl AdversaryWorkersContext {
    pub fn new(exit: Arc<ExitCondition>, thread_hdls: Vec<JoinHandle<()>>) -> Self {
        Self { exit, thread_hdls }
    }

    pub fn join(self) -> thread::Result<()> {
        self.exit.set();
        for hdl in self.thread_hdls {
            hdl.join()?;
        }
        Ok(())
    }
}

pub fn create_rayon_thread_pool(thread_name_prefix: &str) -> Arc<ThreadPool> {
    const MAX_PACKET_FLOOD_THREADS: usize = 8;
    let thread_name_prefix = thread_name_prefix.to_owned();
    Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(get_thread_count().min(MAX_PACKET_FLOOD_THREADS))
            .thread_name(move |i| format!("{thread_name_prefix}{i:02}"))
            .build()
            .unwrap(),
    )
}

pub fn init_keypair_pool(
    keypair_pool: &RwLock<Vec<Keypair>>,
    min_size: usize,
    thread_pool: &ThreadPool,
) {
    const MIN_PARALLEL_ITEMS: usize = 1_000;
    let mut keypair_pool = keypair_pool.write().unwrap();
    if keypair_pool.len() < min_size {
        *keypair_pool = thread_pool
            .install(|| {
                (0..min_size)
                    .into_par_iter()
                    .with_min_len(MIN_PARALLEL_ITEMS)
                    .map(|_i| Keypair::new())
            })
            .collect();
    }
}
