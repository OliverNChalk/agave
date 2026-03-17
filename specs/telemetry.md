# Telemetry: Transaction Pipeline Stamping

## Goal

Per-message timing and outcome tracking across the transaction pipeline.
Every message gets a `seq_id` assigned at ingestion. Stages call into a
lightweight stamper to record recv/send/drop events. Queues are unmodified.

## Stamper

```rust
struct Stamper {
    stage: u8,
    sink: StampSink,
}

impl Stamper {
    fn recv(&self, seq_id: u64);               // action 0x00
    fn send(&self, seq_id: u64);               // action 0x01
    fn drop(&self, seq_id: u64, reason: u8);   // action 0x10+
}
```

Each stage gets a `Stamper`. Queues stay as vanilla shaq — seq_id flows
through existing mechanisms (`M` generic, message fields, etc.).

## Stamps

```rust
/// Emitted to a bounded MPSC channel, drained by a telemetry worker.
#[repr(C)]
struct StageStamp {
    seq_id: u64,
    stage: u8,
    action: u8,
    rdtsc: u64,
}
```

Actions:

| Value | Meaning                                 |
| ----: | --------------------------------------- |
|  0x00 | recv                                    |
|  0x01 | send                                    |
| 0x10+ | drop (application-defined reason codes) |

Derived metrics:

- Queue residence = next stage `recv` rdtsc - previous stage `send` rdtsc
  (same seq_id).
- Processing time = `send` (or `drop`) rdtsc - `recv` rdtsc (same stage,
  same seq_id).

## Enrichment Events

Keyed by `seq_id`, emitted once at the appropriate stage. Not part of the
stamper — emitted by application code to the same telemetry sink.

```rust
struct SignatureMapping {
    seq_id: u64,
    signature: [u8; 64],
}

struct IngestEvent {
    seq_id: u64,
    ingest_ns: i64,   // wall-clock nanos at pipeline entry
    source: u8,        // 0=Tpu, 1=TpuFwd, 2=TpuVote, 3=GossipVote
}
```

Additional enrichment (priority, not_included_reason, etc.) can be added as
separate event types following the same pattern.

## Stages

seq_id is minted at tpu_to_pack (eventually QUIC reception).

| Stage | Queue          | Direction          | Notes                                 |
| ----: | -------------- | ------------------ | ------------------------------------- |
|     0 | tpu_to_pack    | Agave → Scheduler  | 1:1 per packet. seq_id assigned here. |
|     1 | pack_to_worker | Scheduler → Worker | Batched (e.g. bundles). Send stamped per tx. |
|     2 | worker_to_pack | Worker → Scheduler | Batched results. Recv stamped per tx. |

Stages 0–2 cover the scheduling hot path. Pre-scheduler stages (QUIC, fetch,
sigverify) and forwarding can be added later by extending seq_id assignment
upstream.

## seq_id Threading

seq_id flows through existing queue mechanisms, not the stamper:

- **tpu_to_pack**: Added as a field on `TpuToPackMessage`.
- **Scheduler internal state**: Stored alongside transaction in scheduler's
  state map (via `M` generic or `TransactionState`).
- **pack_to_worker / worker_to_pack**: Carried via `M` in
  `KeyedTransactionMeta<M>`.

## Telemetry Worker

Single thread drains `StageStamp` and enrichment events from a bounded MPSC.
Backpressure: if the channel is full, stamps are dropped (never block the hot
path). Sink is pluggable (NATS, file, ring buffer).
