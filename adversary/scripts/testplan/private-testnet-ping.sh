# shellcheck disable=SC2148
#
# Private invalidator testnet configuration and test plan. This script is
# expected to be conditionally sourced by the continuous-test.sh script.
#

set -o errexit
set -o nounset
set -o pipefail

run_attacks_all() {
  test_ping
  test_pingWithPriority
}