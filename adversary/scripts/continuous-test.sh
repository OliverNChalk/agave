#!/usr/bin/env bash
#
# Loop through invalidator test cases with solana-invalidator-client.
# Ctrl-C to exit.
#

set -o errexit
set -o nounset
set -o pipefail

declare me
me=$( realpath "${BASH_SOURCE[0]}" )
declare here
here=$( realpath "$(dirname "${BASH_SOURCE[0]}")" )

#
# === Configuration defaults ===
#
runtime=
sleeptime=
iterations=unbounded
rpcAdversaryKeypair=
attackTarget=
testplan=public-testnet.sh
solanaClient=solana
invalidatorClient=solana-invalidator-client
inimica=solana-invalidator-inimica
accounts=accounts.json

usage() {
  cat <<EOM
Usage:
  $me
    --runtime <secs>
    --sleeptime <secs>
    [--iterations {<num> | unbounded}]
    [--rpc-adversary-keypair <keypair path>]
    [--attack-target {<pubkey> | <IP>}]
    [--invalidator-client <solana-invalidator-client path>]

Configures a running invalidator to run a set of attacks, one after another.
The list of attacks is hardcoded in the scripts.

Arguments:
  --runtime <secs>  Number of seconds to run each attack for.
    Currently, attacks do not specify their duration.  They are executed as long
    as they are active.

  --sleeptime <secs>  Number of seconds to sleep between consequitive attacks.
    The cluster might be affected by an attack for some amount of time after the
    attack has been turned off.  In order to distinguish attack effects more
    clearly in the metrics and logs we wait this long after we disable one
    attack, before we enable the next one.

  --iterations <num>  Number of times to go though the whole attack list.  Use
    "unbounded" to mean, run until stopped with Ctrl+C.
    Optional.

    Default: $iterations

  --rpc-adversary-keypair <keypair path>  File holding a private key, that is
    used to sign RPC calls sent to the invalidator.  Public key is passed into
    the validator on startup using the --rpc-adversary-indentity argument.

    If validator was started with --no-rpc-adversary-auth, then this argument
    should be ommited.

    Default: <empty>

  --attack-target {<pubkey> | <IP>}  Used by repair packet flow tests,
    specifying another validator in the network that will be attacked.

    Default: <empty>

  --testplan <testplan path>  File in the testplan/ directory containing the test
    plan to run.

    Default: $testplan

  --solana-client <solana-client path>  Path to a binary that is the solana client.
    Either absolute or from PATH.  Executed as specified.

    Default: $solanaClient

  --invalidator-client <solana-invalidator-client path>  Path to a binary that
    is the invalidator client.  Either absolute or from PATH.  Executed as
    specified.

    Default: $invalidatorClient

  --inimica <solana-invalidator-inimica path>  Path to a binary that
    is the inimica client.  Either absolute or from PATH.  Executed as
    specified.

    Default: $inimica

  --accounts <accounts JSON file path>  Path to a JSON file that
    contains fee payer account keypairs.  Either absolute or from PATH.

    Default: $accounts
EOM
}

#
# === Argument processing ===
#

die() {
  printf 'ERROR: %s' "$1" >&2
  shift
  [[ $# -gt 0 ]] && printf ' %s' "$@" >&2
  printf '\n' >&2

  exit 1
}

requires_arg() {
  local count=$1
  local value=$2
  local name=$3
  local cliName=$4
  local type=$5
  shift 5
  [[ $# -ne 0 ]] && die "required_arg takes 5 arguments"

  if [[ "$count" -le 0 ]]; then
    die "--$name requires an argument: $type"
  fi

  declare -g "$name"
  printf -v "$name" %s "$value"
}

requires_re_arg() {
  local count=$1
  local value=$2
  local name=$3
  local cliName=$4
  local type=$5
  local expectedRe=$6
  shift 6
  [[ $# -ne 0 ]] && die "required_re_arg takes 6 arguments"

  if [[ "$count" -le 0 ]]; then
    die "--$cliName requires an argument: $type"
  fi

  if [[ ! "$value" =~ $expectedRe ]]; then
    die "--$cliName requires an argument: $type. Got: \"$value\""
  fi

  declare -g "$name"
  printf -v "$name" %s "$value"
}

error() {
  local error=$1

  if [[ -n "$error" ]]; then
    echo "Error: $error"
    echo
  fi

  help_msg

  if [[ -n "$error" ]]; then
    exit 1
  else
    exit 0
  fi
}

while [[ $# -gt 0 ]]; do
  name=$1
  shift

  case $name in
    --runtime)
      requires_re_arg $# "$1" runtime runtime seconds '^[0-9]+$'
      shift
      ;;
    --sleeptime)
      requires_re_arg $# "$1" sleeptime sleeptime seconds '^[0-9]+$'
      shift
      ;;
    --iterations)
      requires_re_arg $# "$1" iterations iterations 'count of "unbounded"' \
        '^([0-9]+|unbounded)$'
      shift
      ;;
    --rpc-adversary-keypair)
      requires_arg $# "$1" \
        rpcAdversaryKeypair "rpc-adversary-keypair" "file path"
      [[ ! -r "$rpcAdversaryKeypair" ]] && \
        die "--rpc-adversary-keypair must point to an existing file." \
          "Got: $rpcAdversaryKeypair"
      shift
      ;;
    --attack-target)
      requires_arg $# "$1" attackTarget "attack-target" "attack name"
      shift
      ;;
    --testplan)
      requires_arg $# "$1" \
        testplan "testplan" "testplan path"
      [[ ! -r "${here}/testplan/${testplan}" ]] && \
        die "--testplan must point to a testplan file in the testplan/ folder." \
          "Got: $testplan"
      shift
      ;;
    --solana-client)
      requires_arg $# "$1" solanaClient "solana-client" \
        "path to binary"
      shift
      ;;
    --invalidator-client)
      requires_arg $# "$1" invalidatorClient "invalidator-client" \
        "path to binary"
      shift
      ;;
    --inimica)
      requires_arg $# "$1" inimica "inimica" \
        "path to binary"
      shift
      ;;
    --accounts)
      requires_arg $# "$1" accounts "accounts" \
        "path to accounts JSON file"
      shift
      ;;
    -h|-\?|--help)
      usage
      exit
      ;;
    *)
      printf 'ERROR: Unexpected argument: "%s"\n\n' "$1" >&2
      usage
      exit 2
      ;;
  esac
done

if ! command -v "$solanaClient" &>/dev/null; then
  cat <<EOM
Unable to find solana client.
Specified path: $solanaClient
Specified via --solana-client.
Default is set in the script header at: $me
EOM
  exit 1
fi

if ! command -v "$invalidatorClient" &>/dev/null; then
  cat <<EOM
Unable to find solana invalidator client.
Specified path: $invalidatorClient
Specified via --invalidator-client.
Default is set in the script header at: $me
EOM
  exit 1
fi

if ! command -v "$inimica" &>/dev/null; then
  cat <<EOM
Unable to find solana inimica client.
Specified path: $inimica
Specified via --inimica.
Default is set in the script header at: $me
EOM
  exit 1
fi

declare -a commonArgs
declare -a repairShArgs

if [[ -z "$runtime" || "$runtime" -eq 0 ]]; then
  die "--runtime argument is required and must be above zero"
fi

if [[ -z "$sleeptime" || "$sleeptime" -eq 0 ]]; then
  die "--sleeptime argument is required and must be above zero"
fi

if [[ -z "$accounts" ]]; then
  die "--accounts argument is required"
fi

if [[ -n "$rpcAdversaryKeypair" ]]; then
  commonArgs+=("--rpc-adversary-keypair" "$rpcAdversaryKeypair")
  repairShArgs+=("--rpc-adversary-keypair" "$rpcAdversaryKeypair")
fi

if [[ -n "$attackTarget" ]]; then
  repairShArgs+=("--attack-target" "$attackTarget")
fi

# shellcheck source=adversary/scripts/invalidator-tests.sh
source "${here}/invalidator-tests.sh"

#
# === Execution ===
#

main() {
  # If RUST_LOG is unset, default to info.
  export RUST_LOG=${RUST_LOG:-solana=info,solana_runtime::message_processor=debug}
  # Useful for debugging client crashes.
  # TODO Do we still want this set?
  export RUST_BACKTRACE=1

  # shellcheck source=adversary/scripts/testplan/public-testnet.sh
  source "${here}/testplan/${testplan}"

  local iter=0
  while true; do
    if [[ "$iterations" != unbounded && "$iter" -ge "$iterations" ]]; then
      break
    fi
    iter=$(( iter + 1 ))

    printf 'Iteration: %s\n' "$iter"
    run_attacks_all
  done
}

main
