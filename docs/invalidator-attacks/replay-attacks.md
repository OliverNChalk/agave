# Replay stage attacks

## Random transfer attack

Creates blocks full of transfer transactions.
Since this type of operation is the computationally simplest, this attack emits blocks with largest number of transactions.

### Objective

Investigate the cluster performance under max possible transfer load.

### Accounts setup

Requires at least 256 payer accounts with sufficient funds.

#### Observations

TODO: Write how to check that the attack has been launched successfully and that the validator performed as expected.

## Chain of transactions

Creates dependency chains of transactions to maximize single thread dependencies.
Since each transactions in one batch must be independent, these chains are between batches of transactions.
Each replay thread has it's own chains.
For example, if number of workers is 1 and size of batch is 2 and we have accounts `[a,b,c,d,e,f,g,h]`, this will create chains `[(a,c), (c,e), (e, g)]` and `[(b,d), (d, f), (f, h)]`.

### Objective

Investigate the effect of chain of transactions on the scheduler performance.

### Accounts setup

Same as for [Random transfer attack](#random-transfer-attack)

#### Observations

TODO: Write how to check that the attack has been launched successfully and that the validator performed as expected.

## Create nonce attack

Creates a large number of nonce accounts as quickly as possible.
Each create nonce transaction uses it's own payer to avoid accounts conflicts.
Fee payer is also nonce account authority.
After the attack, nonce accounts are not cleaned up.

### Objective

Investigate the performance of nonce accounts creation.

### Accounts setup

Same as for [Random transfer attack](#random-transfer-attack)

#### Observations

TODO: Write how to check that the attack has been launched successfully and that the validator performed as expected.

## Allocate numerous small accounts

Creates a block with transactions creating 1 byte accounts.
Similar to previous attack, uses payers accounts to pay the fees.

### Objective

Investigate the performance of small accounts creation along with the effect of such activity on the cluster overall performance.

### Accounts setup

Same as for [Random transfer attack](#random-transfer-attack)

#### Observations

TODO: Write how to check that the attack has been launched successfully and that the validator performed as expected.

## Allocate numerous large accounts

Creates a block with transactions creating 10MB accounts.
Similar to previous attack, uses payers accounts to pay the fees.

### Objective

Investigate the performance of large accounts creation along with the effect of such activity on the cluster overall performance.

### Accounts setup

Same as for [Random transfer attack](#random-transfer-attack)

#### Observations

TODO: Write how to check that the attack has been launched successfully and that the validator performed as expected.

## Max read-only accounts attack

Creates transactions that use maximum number of maximum sized read-only accounts for input.
These transactions do not have any instructions.

### Objective

Overloading the replay machinery IO subsystem.

### Accounts setup

Requires at least 2240 payer accounts with sufficient funds.
The value 2240 comes from `MAX_BATCH_SIZE * TX_MAX_ATTACK_ACCOUNTS_IN_PACKET = 64 * 35`, where the former is how many account addresses can we pack into a single transaction space.

#### Observations

TODO: Write how to check that the attack has been launched successfully and that the validator performed as expected.

## Max accounts attack

Creates transactions that use maximum number of maximum sized accounts for input (not read-only).
These transactions do not have any instructions.

### Objective

Overloading the replay machinery IO subsystem.

### Accounts setup

Same as for [Max read-only accounts attack](#max-read-only-accounts-attack)

#### Observations

TODO: Write how to check that the attack has been launched successfully and that the validator performed as expected.

## Write Program Attack

Constructs a block full of transactions executing WriteProgram [src](https://github.com/solana-labs/invalidator/blob/master/programs/block-generator-stress-test/src/lib.rs#L62).
This program changes the first byte of each account it takes as input to some arbitrary chosen number (128 atm).

### Objective

Triggers the copying of the account data, which determines the computational complexity.

### Attack parameters

* `use_failed_transaction_hotpath` – skip execution of the transactions on the invalidator.
Typically used to avoid replaying transactions that are too computationally demanding (load and execute takes > `400ms`).
To use this option, transactions must be invalid, which might be achieved by using an insufficient `transaction_cu_budget`.

* `transaction_batch_size` – also called "entry size", it determines how many transactions are replayed together within the scope of one replay thread (there are 4 replay threads).
Maximum allowed value is 64.
All the transactions in the batch that don't share account locks are replayed in parallel.

* `num_accounts_per_tx` – how many accounts each transaction modifies.
Note, that limit of accumulative accounts size one transaction can modify is `64MB`.
Also take into account that max account size is `10MB`.

* `transaction_cu_budget` – CU budget for each transaction.
If we create valid transactions and want to have fully packed blocks, be precise with this parameter.

### Accounts setup

Requires:
* deployed program `block_generator_stress_test` which can be found in solana programs folder.
* at least `transaction_batch_size * 4` payer accounts (to avoid `AccountInUse` error)
* at least `transaction_batch_size * num_accounts_per_tx * 4` payer accounts (to avoid `AccountInUse` error)

### Attack setup 1, metrics

```bash
solana-invalidator-client
    -u "http://<invalidator-node>:8899" \
    configure-replay-stage-attack \
    --selected-attack writeProgram \
    --transaction-batch-size 64 \
    --num-accounts-per-tx 1 \
    --transaction-cu-budget 100
```

Use to stress-test network.
Invalidator creates a block that is too heavy to be replayed in the 400ms.
To generate such a block, transactions must fail and, in this case, we can skip execution and loading accounts on the invalidator node (`--use-failed-transaction-hotpath`).

#### Observations

 * Validator is skipping blocks when the invalidator is leader (use mean tx cost per node) to identify these events:

```
SELECT "slot" - "parent_slot" as "diff_slot", slot  FROM :DB_NAME:."autogen"."bank-new_from_parent-heights" WHERE time > :dashboardTime: AND time < :upperDashboardTime: FILL(null)
```

* Transaction cost will be high during block replay:
```
SELECT mean("block_cost") / mean("transaction_count") AS "mean_CU_PER_Tx" FROM :DB_NAME:."autogen"."cost_tracker_stats" WHERE time > :dashboardTime: AND time < :upperDashboardTime: GROUP BY time(:interval:), "host_id" FILL(null)
```

### Attack setup 2, metrics

```bash
solana-invalidator-client
    -u "http://<invalidator-node>:8899" \
    configure-replay-stage-attack \
    --selected-attack writeProgram \
    --transaction-batch-size 1 \
    --num-accounts-per-tx 1 \
    --transaction-cu-budget 20000
```

Use to determine how many such transactions in one block we can replay.
Contrary to the previous setup, invalidator will create blocks with all valid transactions that will be replayed on invalidator node as well.

#### Observations

* very few transactions will land to the block (40-60 per block).
To manually check this, use https://explorer.solana.com/.
For that, open ssh tunnel to the target node `ssh -L 8899:127.0.0.1:8899 -L 8900:127.0.0.1:8900 <invalidator-node>`.
In the explorer, select localhost and search for the `program_id` of the deployed program.

## Read Program Attack

Construct a block which contains transactions each reading provided number of large accounts (10MB).

### Objective

Evaluate the effect of reading large data in on-chain program.

### Attack parameters

Same as for [Write program attack](#write-program-attack).

### Accounts setup

Same as for [Write program attack](#write-program-attack).

### Attack setup, metrics

```bash
solana-invalidator-client
    -u "http://<invalidator-node>:8899" \
    configure-replay-stage-attack \
    --selected-attack readProgram \
    --transaction-batch-size 64 \
    --num-accounts-per-tx 1 \
    --transaction-cu-budget 1000
```

#### Observations

TODO

## Recursive Program Attack

Construct a block which contains transactions each calling recursive program which does nothing.

### Objective

Evaluate the effect of cross program invocations on the performance of the cluster.
CPI triggers copying of all the accounts involved leading to high load on the replay stage.

### Attack parameters

Same as for [Write program attack](#write-program-attack).

### Accounts setup

Same as for [Write program attack](#write-program-attack).

### Attack setup, metrics

```bash
solana-invalidator-client
    -u "http://<invalidator-node>:8899" \
    configure-replay-stage-attack \
    --selected-attack recursiveProgram \
    --transaction-batch-size 64 \
    --num-accounts-per-tx 1 \
    --transaction-cu-budget 2000
```

#### Observations

TODO

## JIT Cache Attack

Construct a block which contains transactions each trying to execute a unique large program.
Each program call is invalid because we call some big program and pass not enough accounts.
But each of these programs will be compiled independently and will be be added to cache.

### Objective

Stress test validators replay stage by loading jit cache.
During this attack the cluster performance is very low.
It allows to investigate performance of compilation and jit cache implementation.

### Attack parameters

Same as for [Write program attack](#write-program-attack).

### Accounts setup

Requires:
* at least `transaction_batch_size * 4` payer accounts (to avoid `AccountInUse` error)
* at least 512 keypair files to be used to deploy a program. 512 comes from `PROGRAM_CACHE_SIZE * 2`.

### Attack setup, metrics

```bash
solana-invalidator-client
    -u "http://<invalidator-node>:8899" \
    configure-replay-stage-attack \
    --selected-attack coldProgramCache \
    --transaction-batch-size 64 \
    --num-accounts-per-tx 1 \
    --transaction-cu-budget 20000 \
    --use-failed-transaction-hotpath
```

#### Observations

* Peaks of programs cache hit/misses showing close values during the attack

```
SELECT mean("hits") AS "mean_hits", mean("misses") AS "mean_misses" FROM :DB_NAME:."autogen"."loaded-programs-cache-stats" WHERE time > :dashboardTime: AND time < :upperDashboardTime: GROUP BY time(:interval:) FILL(null)
```

* Replay batch time is high and this cannot be explained only by `execute_us`. The same was observed on mainnet 1 Dec.

```
SELECT mean("execute_batches_us") AS "mean_execute_batches_us", mean("execute_details_execute_inner_us") AS "mean_execute_details_execute_inner_us", mean("execute_us") AS "mean_execute_us" FROM :DB_NAME:."autogen"."replay-slot-stats" WHERE time > :dashboardTime: AND time < :upperDashboardTime: GROUP BY time(:interval:) FILL(null)
```

* From `create_executor_trace` metric (requirece `TRACE` log level), programs are added to cache not uniformly, time to compile has a very long tail of distribution and is not normal.
