use {
    rayon::{prelude::*, ThreadPool},
    solana_keypair::Keypair,
    solana_rayon_threadlimit::get_thread_count,
    std::{
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc, RwLock,
        },
        thread::{self, JoinHandle},
    },
};

#[derive(Debug)]
pub struct AdversaryWorkersContext {
    exit: Arc<AtomicBool>,
    thread_hdls: Vec<JoinHandle<()>>,
}

impl AdversaryWorkersContext {
    pub fn new(exit: Arc<AtomicBool>, thread_hdls: Vec<JoinHandle<()>>) -> Self {
        Self { exit, thread_hdls }
    }

    pub fn join(self) -> thread::Result<()> {
        self.exit.store(true, Ordering::Relaxed);
        for hdl in self.thread_hdls {
            hdl.join()?;
        }
        self.exit.store(false, Ordering::Relaxed);
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
