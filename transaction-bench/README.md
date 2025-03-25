## Client to stress-test validator

Client application that sends transactions using QUIC protocol to the TPU port.
Firstly, current tool creates accounts necessary for the transaction generation.
Secondly, it generates transactions executing on-chain program  and sends
them to the upcoming leaders.
Currently used on-chain program can be found in `programs/client-test-program`.
Contrary to other stress-testing tools, the current one doesn't use `TpuClient` and `ConnectionCache` to
interact with validators but uses alternative client network layer implementation.

### To run with solana-test-validator

1. Build and deploy program
  * On a cluster with loader v4 support.
```shell
pushd programs/client-test-program > /dev/null
cargo build-sbf
popd > /dev/null
solana -ul program-v4 deploy target/deploy/client_test_program.so \
  --program-keypair target/deploy/client_test_program-keypair.json
```

To redeploy a program, you must use '--program-id' instead of '--program-keypair'.
```shell
solana -ul program-v4 deploy target/deploy/client_test_program.so \
  --program-id \<program_id\>
```

  * On a cluster with loader v3 support.

Same command for initial program deployment and redeployment.
```shell
pushd programs/client-test-program > /dev/null
cargo build-sbf
popd > /dev/null
solana -ul program deploy target/deploy/client_test_program.so \
  --program-id target/deploy/client_test_program-keypair.json
```

2. Run the tool:

```shell
args=(
  -u "$ENDPOINT"
  --authority "$HOME/$ID_FILE"
  --staked-identity-file "$HOME/$CLIENT_NODE_ID_FILE"
  --duration "$DURATION"
  --num-payers 256
  --num-accounts 1024
  --account-size "[128,512]"
  --payer-account-balance 1
  --account-owner ${PROGRAM_ID}
  --num-accounts-per-tx "[1,4]"
  --transaction-cu-budget 600
  #--validate-accounts -- if you want to check that the program has been deployed and accounts have been successfully created.
  #--pinned-address "${CLIENT_NODE_IP}:8009" -- if you want to send txs to exactly this peer
)
solana-transaction-bench "${args[@]}" 2> err.txt
```

### TODO

* Implement program deployment as part of the client code to simplify the process
* Change sending packets with streams: use one stream per connection.
* Calculate num streams per second.
* Add support of many programs.
* Add reading/writing accounts from file. It is important for tesnet.
* Implement multi-stake identities support. This change will require using more than one connection for the peer (each having it's own identity).
* Use 0rtt connection when necessary. For private cluster not really an issue.
